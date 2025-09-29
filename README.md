# Tesi-Magistrale-SL
## Repository structure

```text

.
├─ scripts/       
│  ├─ eval/
│  │  ├─ barplot.py                           # grafici distribuzioni/emozioni e confronto metriche
│  │  ├─ eval_MobileNetV2_4.py                # evaluation su test / sottogruppi
│  │  ├─ eval_and_save.py                     # valutazione modelli multipli + salvataggio CSV metriche
│  │  ├─ gradcam.mobilenetv2.py               # visualizzazioni Grad-CAM
│  │  ├─ validation.py                        # griglie di campioni e check visivo delle annotazioni
│  │  └─ README.md
│  ├─ preprocessing/                          # selezione casi Medium
│  │  ├─ extract_medium.py                    # estrazione casi medium per la selezione manuale
│  │  ├─ generate_splits_4.py                 # split stratificati (emotion × skin)
│  │  ├─ merge_binary.py                      # colonna HL_binary (Light/Dark)
│  │  ├─ prepare_dataframe.py                 # build CSV con path/etichette RAF-DB
│  │  ├─ review_medium.py                     # etichettatura manuale Medium→Light/Dark
│  │  ├─ skin_tone_batch.py                   # stima skin tone (Light/Medium/Dark)
│  │  ├─ diagnostic.py                        # filtri QC (grayscale/low-color)
│  │  └─ README.md
│  ├─ tools/
│  │  ├─ backbone.py                          # summary of MobileNetV2 backbone
│  │  ├─ extract_bottlenecks.py               # estrazione feature (bottleneck) per training veloce/ablation
│  │  ├─ inspect_backbone.py                  # ispezione strati/nome layer del modello caricato 
│  │  └─ README.md
│  ├─ train/
│  │  ├─ MobileNetV2_4.py                     # training baseline + fine-tuning
│  │  ├─ mitig_equalized_odds_postproc.py     # mitigation using Equalized Odds (post-processing)
│  │  ├─ mitigation_adv1.py                   # mitigation using Adversarial Debiasing (in-process)
│  │  ├─ mitigation_oversampling3.py.py       # mitigation using Oversampling (pre-processing)
│  │  ├─ mitigation_reweight2.py              # mitigation using Reweight (pre-processing)
│  │  └─ README.md
├─ gitignore                                      
└─ README.md
