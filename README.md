# Yolo-V8-Tuning
YoloV8 Tuning

# Install
```bash
pip install -r requirements.txt
```

# Run
## Required Files
### Checkpoints
Make sure to have your respective checkpoints in the checkpoints directory.
Change paths of checkpoints to your respective paths in main.py
### Datasets
Make sure to have your respective datasets somewhere in the project directory.
Change paths of datasets to your respective paths in main.py

## Run the file
Run main.py
```bash
python main.py --help
```

You will need to specify arg parser parameters.

### Training
```bash
python main.py --train "cfg/drone_train.yaml"
```

## Debug
Debugger `launch.json` setup for vscode. Feel free to modify
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Coco128 Train",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/code/main.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--train",
                "${workspaceFolder}/code/cfg/coco128_train.yaml"
            ],
            "cwd": "${workspaceFolder}/code/"
        },
        {
            "name": "Drone Real Train",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/code/main.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--train",
                "${workspaceFolder}/code/cfg/drone_train.yaml"
            ],
            "cwd": "${workspaceFolder}/code/"
        }
    ]
}
```
