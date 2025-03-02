
## Quick Start
models link
https://nusu-my.sharepoint.com/personal/e1330352_u_nus_edu/Documents/Forms/All.aspx?RootFolder=%2Fpersonal%2Fe1330352%5Fu%5Fnus%5Fedu%2FDocuments%2FISY5004%20Practice%20Module%2FModels%2Fmodels&FolderCTID=0x0120009F0412ECCE1BA747B6FEF808642CE9C5&View=%7B6D635FE0%2D2222%2D41BF%2DAD8A%2DA505EE1B97A8%7D

Download the requirements file first inside the PhotoLab, and also download some extra packages:
Python 3.12
pip install requirement
```console

pip install PyQt6
pip install pyqtgraph
pip install panda3d
```

Download the pretrained models by running the included download script:

```console
foo:bar$ python download_models.py
```

Start the editor by running:

```console
foo:bar$ python src/main.py
```

If you face any issue relate to "qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem", here is several methods you could try to figure it out:

First, try reinstall PyQt6 PyQt6-Qt6 packages:

```console
pip install --force-reinstall PyQt6 PyQt6-Qt6
```

Second, set the export route for plugin:

```console
export QT_PLUGIN_PATH="/Users/liqi/anaconda3/lib/python3.12/site-packages/PyQt6/Qt6/plugins"
```

After all these steps, try rerun the "python src/main.py" in your terminal.

##UI TODO

1. Selection Tool & Magic Wand-like Function
Implement a basic selection tool and a magic wand-like function to select objects within an image.
The selection tool should allow users to move, rotate, and scale the selected object.
High precision is not required, as AI will handle auto-selection in the future.
Ensure that the selection tool provides basic usability and interaction.

2. Layer Management
Provide basic layer functionality, including adding, removing, flatten layer

3. Main Toolbar for File Operations
Add a main toolbar with essential file operations:
Open File (Ctrl + O)
Save File (Ctrl + S)
Undo Action (Ctrl + Z)
Ensure that these shortcut keys function correctly.

## Features
- NIMA Score (Need to Impleemnt)
- Instagram Filters
- Super-Resolution
- Human Segmentation (Need to Intergrate)
- Inpainting (Need to Implement)




