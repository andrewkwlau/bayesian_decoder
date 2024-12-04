# Bayesian Decoder

This is my dissertation project for my MScR in Integrative Neuroscience at the University of Edinburgh.

> Does mouse V1 cortex encode spatial representations in light and in dark (where visual cues are absent)?

To investigate this, members of the Rochefort Lab utilised a VR navigation task and record mice V1 activity with 2-photon calcium imaging while the mice were running in the virtual tunnel:

![experimental_setup](./assets/experimental_setup.png)

As part of this bigger project, I use Bayesian inference to decode the spatial location of mice from their V1 neural activity. This offers preliminary evidence that the mice V1 encode spatial representations both in light and in dark where visual cues are absent.

Full dissertation:
[here](https://drive.google.com/file/d/1aufyHlR_6vslDHpS80ekPQCx5uOTfaKf/view?usp=sharing) (results are not the most updated)

## Demonstration and Results
You can also see a demo of the decoding pipeline
[here](scripts/demo.ipynb).


### Decoder Accuracy
![accuracy](./assets/savefig/accuracy_all.png)

### Decoder Errors
![errors](./assets/savefig/errors_all.png)

### Sample confusion matrix in light
![confusion_mtx_lgtlgt](./assets/savefig/C57_60_Octavius_confusion_mtx_lgtlgt.png)

### Sample confusion matrix in dark
![confusion_mtx_drkdrk](./assets/savefig/C57_60_Octavius_confusion_mtx_drkdrk.png)