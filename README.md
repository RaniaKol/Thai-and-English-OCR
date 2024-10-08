# Thai-and-English-OCR

This repository was created for submitting the first assignment for the course Machine Learning for Statistical NLP: Advanced (LT2326).
All instructions on how to run the scripts exist below. 
On the .pdf file there is the description of the experiments that I run.
To run the scripts:
1. python3 data.py --lang Thai English --dpi 200 300 --style bold --tr_ratio 0.8 --te_ratio 0.1 --val_ratio 0.1 --input_path path to/ThaiOCR/ThaiOCR-TrainingSet --output_path path to output folder
2. python3 train.py --tr_set /path to /training_set.txt --val_set /path to/validation_set.txt --batch 32 --epochs 10 --save_dir /path to directory for saving the model --log_file / path to/logs.log
3. python3 eval.py --test_set /path to/testing_set.txt --model_path /path to/model.pth --batch 32