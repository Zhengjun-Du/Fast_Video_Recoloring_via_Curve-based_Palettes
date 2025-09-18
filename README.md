## Fast Video Recoloring via Curve-based Palettes

![](https://github.com/Zhengjun-Du/Fast_Video_Recoloring_via_Curve-based_Palettes/blob/main/teaser.png) 

This is the source code of the paper: **Fast Video Recoloring via Curve-based Palettes**, authors: Zheng-Jun Du, Jia-Wei Zhou, Kang Li, Jian-Yu Hao, Zi-Kang Huang, Kun Xu*. Please fell free to contact us if you have any questions, email: dzj@qhu.edu.cn

### Requirements

1. Windows 10  
2. Microsoft Visual Studio 2019 or higher version
3. OpenCV 4.1
4. Nlopt 2.4.2
5. Qt 5.12.12

### Directories

1. data: a video for test
2. libs: OpenGL, NLopt
3. Qt-Color-Widgets：color Picker

### Usage

1. Click "Open Video" to load the video.

2. Click "Extract Palette" to generate the original color palette. You can edit the color palette by selecting key frames using the progress bar in "Video progress".

3. After editing, click “Recolor” to recolor the video.

4. Other features:

   a) click "Reset" to restore all palettes. You can also right-click on the corresponding palette to reset its colors individually.

   b) click "Import Palette", "Export Palette" and "Export Video" to import the color palette, export the color palette, and export the video, respectively. Imported and exported data are stored in "./files.txt".

![](https://github.com/Zhengjun-Du/Fast_Video_Recoloring_via_Curve-based_Palettes/blob/main/GUI.png) 

### References
[1] Du Z J, Zhou J W, Li K, Hao J Y, Huang Z K, Xu K. Fast Video Recoloring via Curve-based Palettes[J], IEEE TIP, 2025.

[2] Du Z J, Lei K X, Xu K, et al. Video recoloring via spatial-temporal geometric palettes[J]. ACM Trans. Graph., 2021, 40(4): 150:1-150:16.

[3] H. Chang, O. Fried, Y. Liu, S. DiVerdi, and A. Finkelstein, “Palette-based photo recoloring.” ACM Trans. Graph., vol. 34, no. 4, pp. 139–1, 2015.

[4] J. Tan, J.-M. Lien, and Y. Gingold, “Decomposing images into layers via RGB-space geometry,” ACM Transactions on Graphics (TOG), vol. 36, no. 1, pp. 1–14, 2016.
