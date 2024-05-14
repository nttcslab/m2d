"""Main program for the paper: Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection
"""

import sys
sys.path.append('../heart-murmur-detection')
sys.path.append('../heart-murmur-detection/ModelEvaluation')

from evar.common import (sys, np, pd, kwarg_cfg, Path,
    torch, logging, append_to_csv, RESULT_DIR)
import torchaudio
import fire

from evar.data import create_dataloader
import evar
from lineareval import make_cfg
from finetune import TaskNetwork, finetune_main

from DataProcessing.find_and_load_patient_files import load_patient_data
from DataProcessing.helper_code import load_recordings
from ModelEvaluation.evaluate_model import evaluate_model
from tqdm import tqdm


def infer_and_eval(cfg, model, test_root, eval_mode='follow_prior_work'):
    model.eval()
    
    pids = sorted(list(set([f.stem.split('_')[0] for f in Path(test_root).glob('*.wav')])))  # evaluate_model.py::find_challenge_files -> sorted(os.listdir(label_folder))
    txt_files = [test_root+pid+'.txt' for pid in pids]
    print('Test file folder:', test_root)
    print('Test files:', pids[:2], txt_files[:2])
    softmax_fn = torch.nn.Softmax(dim=1)
    probabilities, wav_probabilities = [], []

    for txt in tqdm(txt_files):
        # Load recordigns
        data = load_patient_data(txt)
        recordings, frequencies = load_recordings(test_root, data, get_frequencies=True)
        recordings = [torch.tensor(r / 32768.).to(torch.float) for r in recordings]

        # Note: No normalization of raw audio wave. Already normalized in the pipeline.
        #   recordings[0].max() -> tensor(1.0000)
        #   recordings[0].min() -> tensor(-1.)
        # def normalize(wav):
        #     return wav / (1.0e-10 + wav.abs().max())
        # recordings = [normalize(r) for r in recordings]

        wavs = [torchaudio.transforms.Resample(f, cfg.sample_rate)(r) for r, f in zip(recordings, frequencies)]

        # Note: *No padding* because sample lengths are very different among recordings, for example: [164608, 150272, 105472, 460544]
        # print([len(w) for w in wavs])
        # max_len = max([len(w) for w in wavs])
        # wavs = [(np.pad(w, (0, max_len - len(w)) if len(w) < max_len else w) for w in wavs)]

        # Process per recording (with variable length)
        L = cfg.unit_samples  # number of samples for 5 sec
        logits = []
        for wav in wavs:
            if len(wav) < L:
                wav = torch.nn.functional.pad(wav, (0, L - len(wav)))
            # Split wav into 5-s segments and encode them.
            segment_logits = []
            for widx, pos in enumerate(range(0, len(wav) - L + 1, L)):
                segment = wav[pos:pos+L]
                if len(segment) < L:
                    continue
                with torch.no_grad():
                    x = segment.unsqueeze(0)
                    logit = model(x)
                segment_logits.append(logit)       # [1, 3] for one chunk
            # Logits for one recording wav.
            logits.append(torch.stack(segment_logits).mean(0))

        # Reorder classes from ["Absent", "Present", "Unknown"] -> ["Present", "Unknown", "Absent"]
        logits = torch.vstack(logits)
        logits = logits[:, [1, 2, 0]]
        # Probabilities for each wav
        probs = logits.softmax(1).detach().to('cpu')
        wav_probabilities.append(probs)
        # Probability for the average logits
        probs = logits.mean(0, keepdims=True).softmax(1).detach().to('cpu')[0]
        probabilities.append(probs)

    probabilities = torch.stack(probabilities)
 
    def label_decision_rule(wav_probs):
        # Following Panah et al. “Exploring Wav2vec 2.0 Model for Heart Murmur Detection.” EUSIPCO, 2023, pp. 1010–14.
        cidxs = torch.argmax(wav_probs, dim=1)
        PRESENT, UNKNOWN, ABSENT = 0, 1, 2
        # - Assign present if at least one recording was classified as present.
        if PRESENT in cidxs:
            final_label = PRESENT
        # - Assign unknown if none of the recordings was classified as present, and at least one recording was classified  as unknown.
        elif UNKNOWN in cidxs:
            final_label = UNKNOWN
        # - Assign absent if all recordings were classified as absent.
        else:
            final_label = ABSENT
        return final_label

    if eval_mode is None or eval_mode == 'follow_prior_work':
        print('Label decision follows: Panah et al. “Exploring Wav2vec 2.0 Model for Heart Murmur Detection.” EUSIPCO, 2023, pp. 1010–14.')
        cidxs = torch.tensor([label_decision_rule(wav_probs) for wav_probs in wav_probabilities])
    elif eval_mode == 'normal':
        print('Label decision is: torch.argmax(probabilities, dim=1)')
        cidxs = torch.argmax(probabilities, dim=1)
    else:
        assert False, f'Unknown eval_mode: {eval_mode}'
    labels = torch.nn.functional.one_hot(cidxs, num_classes=3)

    wav_probabilities = [p.numpy() for p in wav_probabilities]
    probabilities = probabilities.numpy()
    labels = labels.numpy()
    return evaluate_model(test_root, probabilities, labels), (wav_probabilities, probabilities)


def eval_main(config_file, task, checkpoint, options='', seed=42, lr=None, hidden=(), epochs=None, early_stop_epochs=None, warmup_epochs=None,
              mixup=None, freq_mask=None, time_mask=None, rrc=None, training_mask=None, batch_size=None,
              optim='sgd', unit_sec=None, verbose=False, data_path='work', eval_mode=None, save_prob=None):
    
    cfg, n_folds, balanced = make_cfg(config_file, task, options, extras={}, abs_unit_sec=unit_sec)
    lr = lr or cfg.ft_lr
    cfg.mixup = mixup if mixup is not None else cfg.mixup
    cfg.ft_early_stop_epochs = early_stop_epochs if early_stop_epochs is not None else cfg.ft_early_stop_epochs
    cfg.warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg.warmup_epochs
    cfg.ft_epochs = epochs or cfg.ft_epochs
    cfg.ft_freq_mask = freq_mask if freq_mask is not None else cfg.ft_freq_mask
    cfg.ft_time_mask = time_mask if time_mask is not None else cfg.ft_time_mask
    cfg.ft_rrc = rrc if rrc is not None else (cfg.ft_rrc if 'ft_rrc' in cfg else False)
    cfg.training_mask = training_mask if training_mask is not None else (cfg.training_mask if 'training_mask' in cfg else 0.0)
    cfg.ft_bs = batch_size or cfg.ft_bs
    cfg.optim = optim
    cfg.unit_sec = unit_sec
    cfg.data_path = data_path

    train_loader, valid_loader, test_loader, multi_label = create_dataloader(cfg, fold=n_folds-1, seed=seed, batch_size=cfg.ft_bs,
        always_one_hot=True, balanced_random=balanced)
    print('Classes:', train_loader.dataset.classes)
    cfg.eval_checkpoint = checkpoint

    cfg.runtime_cfg = kwarg_cfg(lr=lr, seed=seed, hidden=hidden, mixup=cfg.mixup, bs=cfg.ft_bs,
                                freq_mask=cfg.ft_freq_mask, time_mask=cfg.ft_time_mask, rrc=cfg.ft_rrc, epochs=cfg.ft_epochs,
                                early_stop_epochs=cfg.ft_early_stop_epochs, n_class=len(train_loader.dataset.classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make a fresh model
    ar = eval('evar.'+cfg.audio_repr)(cfg).to(device)
    if hasattr(train_loader, 'lms_mode') and train_loader.lms_mode:
        ar.precompute_lms(device, train_loader)
    else:
        ar.precompute(device, train_loader)
    task_model = TaskNetwork(cfg, ar).to(device)
    task_model_dp = torch.nn.DataParallel(task_model).to(device)
    # Load checkpoint
    print('Using checkpoint', checkpoint)
    print(task_model_dp.load_state_dict(torch.load(checkpoint, map_location=device)))
    task_model_dp.eval()

    circor_no = task[-1]  # ex) '1' of 'circor1'
    stratified_data = f'../heart-murmur-detection/data/stratified_data{circor_no}/test_data/'
    results, probs = infer_and_eval(cfg, task_model_dp, stratified_data, eval_mode=eval_mode)
    (   classes,
        auroc,
        auprc,
        auroc_classes,
        auprc_classes,
        f_measure,
        f_measure_classes,
        accuracy,
        accuracy_classes,
        weighted_accuracy,
        uar,
    ) = results

    name  = f'{cfg.id}{"" if cfg.weight_file != "" else "/rnd"}-'
    report = f'Finetuning {name} on {task} -> weighted_accuracy: {weighted_accuracy:.5f}, UAR: {uar:.5f}, recall per class: {accuracy_classes}'
    report += f', best weight: {checkpoint}, config: {cfg}'
    logging.info(report)

    result_df = pd.DataFrame({
        'representation': [cfg.id.split('_')[-2]], # AR name
        'task': [task],
        'wacc': [weighted_accuracy],
        'uar': [uar],
        'r_Present': [accuracy_classes[0]],
        'r_Unknown': [accuracy_classes[1]],
        'r_Absent': [accuracy_classes[2]],
        'weight_file': [cfg.weight_file],
        'run_id': [cfg.id],
        'report': [report],
    })
    csv_name = {
        None: 'circor-scores.csv',
        'follow_prior_work': 'circor-scores.csv',
        'normal': 'circor-scores-wo-rule.csv',
    }[eval_mode]
    append_to_csv(f'{RESULT_DIR}/{csv_name}', result_df)

    if save_prob is not None:
        for i, var in zip(['_1', '_2'], probs):
            prob_name = Path(save_prob)/str(checkpoint).replace('/', '-').replace('.pth', i + '.npy')
            #probs = [p.numpy() for p in probs]
            prob_name.parent.mkdir(parents=True, exist_ok=True)
            np.save(prob_name, np.array(var, dtype=object))
            print('Probabilities saved as:', prob_name)


def finetune_circor(config_file, task, options='', seed=42, lr=None, hidden=(), epochs=None, early_stop_epochs=None, warmup_epochs=None,
                  mixup=None, freq_mask=None, time_mask=None, rrc=None, training_mask=None, batch_size=None,
                  optim='sgd', unit_sec=None, verbose=False, data_path='work', eval_only=None, eval_mode=None, save_prob='probs'):

    assert task in [f'circor{n}' for n in range(1, 3+1)]

    # We train a model using the original fine-tuner from the EVAR (finetune_main), and the best_path holds the path of the best weight.
    # This part is the same training process as what we have been doing in BYOL-A and M2D.
    if eval_only is None:
        report, scores, best_path, name, cfg, logpath = finetune_main(config_file, task, options=options, seed=seed, lr=lr, hidden=hidden, epochs=epochs,
            early_stop_epochs=early_stop_epochs, warmup_epochs=warmup_epochs,
            mixup=mixup, freq_mask=freq_mask, time_mask=time_mask, rrc=rrc, training_mask=training_mask, batch_size=batch_size,
            optim=optim, unit_sec=unit_sec, verbose=verbose, data_path=data_path)
        del report, scores, name, cfg, logpath
    else:
        best_path = eval_only

    # Then, we evaluate the trained model specifically for the CirCor problem setting.
    return eval_main(config_file, task, best_path, options=options, seed=seed, lr=lr, hidden=hidden, epochs=epochs,
        early_stop_epochs=early_stop_epochs, warmup_epochs=warmup_epochs,
        mixup=mixup, freq_mask=freq_mask, time_mask=time_mask, rrc=rrc, training_mask=training_mask, batch_size=batch_size,
        optim=optim, unit_sec=unit_sec, verbose=verbose, data_path=data_path, eval_mode=eval_mode, save_prob=save_prob)


if __name__ == '__main__':
    fire.Fire(finetune_circor)
