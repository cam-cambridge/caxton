# CAXTON Data set

The CAXTON data set is a large scale, optical, in-situ process monitoring data set for extrusion AM containing over 1 million sample images of material deposition from the printer nozzle. Each image is labelled with the printing process parameters being used as the image was captured. The data set is highly diverse comprising 192 2D and 3D geometries printed in a wide range of colours from PLA feedstock.

## About Data collection methodology

A fleet of 8 thermoplastic extrusion 3D printers (Creality CR-20 Pro) equipped with cameras (Logitech C270) focused on the nozzle tip were used to collect the data. These machines made use of a unique data generation and collection workflow which was developed to autonomously acquire and label the images with no human input. An outline of this workflow is as follows:

A range of STL files were downloaded from online repositories with a Python-based scraping tool. Toolpaths of  these STL files were subsequently generated using pseudo-random slicing parameters for increased diversity. Numerous settings such as rotation, scale, infill pattern, infill density, perimeter count, and more were sampled from uniform distributions to achieve this randomised slicing. The generated G-code files were then split to have maximum move lengths of no more than 2.5mm - this was to reduce response times when updating printing parameters during a print. A different split file was sent to each printer in the network for printing. During the printing process a set of printing parameter combinations were sampled from uniform distributions for each printer in the fleet. Specifically these parameters were flow rate, lateral speed, z offset, and hotend temperature. The printer then collected 150 images for that combination of parameters, after which a new set were sampled and another 150 images captured. This process continued until the print was completed. Upon print completion, an automatic part remover removed the finished part from the print bed so that the next print could be started, in turn enabling continuous operation.

### Description of the data

An outline of the data set structure can be seen below. It primarily consists of image files and CSVs containing the labels for each image. Inside the root directory there are 192 subdirectories - `print0-191` - one for each print completed. Inside each subdirectory there are images - `image-0-N.jpg` - where `N` is the total number of images for that print alongside a CSV - `print_log_full.csv` - containing the labels for the images. Prints numbering `print183-191` contain large scale errors or poor lightning and were not used for the original purpose of printing parameter prediction. However, the data for these prints may be useful to train detects to classify large scale errors from images of nozzles. As such the file `print_log_filtered.csv` is only found in the subdirectories for prints `print0-182`. This filtered file removes unwanted data where parameters were stored incorrectly and out of their desired ranges. The file `print_log_filtered_classification3.csv` contains extra columns for binning each parameter into 3 levels corresponding to low, good, and high states. Finally, `print_log_filtered_classification3_gaps.csv` removes 15 images (approximately 6 seconds worth of printing) after each parameter change. This is due to a combination of software and mechanical delays resulting in a deplay until a parameter update is visible in an image and in steady state condition.

```
dataset/
  -README.md
  -caxton_dataset_full.csv
  -caxton_dataset_filtered.csv
  -caxton_dataset_filtered_no_outliers.csv
  -caxton_dataset_filtered_no_outliers_img_info.csv
  -caxton_dataset_final.csv
  -print0/
    -print_log_full.csv
    -print_log_filtered.csv
    -print_log_filtered_classification3.csv
    -print_log_filtered_classification3_gaps.csv
    -image-0.jpg
    -image-1.jpg
    -image-2.jpg
    -image-3.jpg
    -...
  -print1/
    -print_log_full.csv
    -print_log_filtered.csv
    -print_log_filtered_classification3.csv
    -print_log_filtered_classification3_gaps.csv
    -image-0.jpg
    -image-1.jpg
    -image-2.jpg
    -image-3.jpg
    -...
  -...
```

### And file formats

```
-1 README.md
-1272465 images, format JPG.
-746 CSV files.
```

Of the 1272465 images, 1272273 have full labels in the CSV files. The `print_log_full.csv` file for each print contains the following information:

* img_path - *path to the respective image in the format* `caxton_dataset/printX/image-Y.jpg`.
* timestamp - *timestamp for when the image was taken in the format* `YYYY-MM-DDTHH:mm:ss-SS`.
* flow_rate - *relative flow rate as a percentage (%).*
* feed_rate - *relative feed rate / lateral speed as a percentage (%).*
* z_offset - *nozzle Z offset in millimetres (mm); negative is lower, positive is higher.*
* target_hotend - *requested hotend temperature in degrees celsius (°C).*
* hotend - *measured hotend temperature using thermistor in degrees celsius (°C).*
* bed - *measured print bed temperature using thermistor in degrees celsius (°C).*
* nozzle_tip_x - *X axis pixel / coordinate location of the nozzle tip in the image.*
* nozzle_tip_y - *Y axis pixel / coordinate location of the nozzle tip in the image.*
* print_id - *unique ID for print - matches the number in the subdirectory name.* 

Here are the first 3 lines from the `print0/print_log_full.csv` file as an example.

```
img_path,timestamp,flow_rate,feed_rate,z_offset,target_hotend,hotend,bed,nozzle_tip_x,nozzle_tip_y,print_id
caxton_dataset/print0/image-1.jpg,2020-10-08T13:12:48-02,100,100,0.0,205.0,204.34,65.66,531,554,0
caxton_dataset/print0/image-2.jpg,2020-10-08T13:12:48-48,100,100,0.0,205.0,204.34,65.66,531,554,0
caxton_dataset/print0/image-3.jpg,2020-10-08T13:12:48-94,100,100,0.0,205.0,204.13,65.74,531,554,0
...
```

### Further information

Each image in the data set is full RGB and 1280x720 pixels in resolution. For intial parameter prediction work the image was cropped to a 320x320 square region centered on the provided nozzle tip coordinates for each images. This region was then resized to 224x224 for inference in the developed deep learning model. The images are all taken with the same model of camera; however, can vary considerably depending on lightning conditions and the lateral speed of the print head during capturing.

#### Flow rate

This data consists of relative percentage values from optimal (at 100%). The labelling of these values is very reliable. However, it is important to note that there may be a small unquantified systematic offset error due to a range of possible optimal flow rate levels (e.g. 99% flow is nearly identical to 100% flow). This was mitigated by calibrating every printers hotend and extruder, and through using appropriate toolpath generation and slicing settings. 

The data distribution is approximately uniform with some added noise, with the exception of at 100% where more data was collected. The data predominantly falls in the range of 20% - 200%.

#### Feed rate / lateral speed

This data consists of relative percentage values from optimal (at 100%). The labelling of these values is very reliable. However, it is important to note that there may be a unquantified systematic offset error due to a range of possible optimal feed rate levels (prints can succeed at a range of different speeds - a 30mm/s print will look very similar to a 35mm/s print). The definition of optimal is more flexible than for flow rate and therefore the error may be slightly larger. This was mitigated by calibrating every printers speed beforehand, and through using appropriate toolpath generation and slicing settings.

The data distribution is approximately uniform with some added noise, with the exception of at 100% where more data was collected. The data predominantly falls in the range of 20% - 200%.

#### Z offset

This is the most noisy labelled parameter. Very small differences in Z offset can lead to large changes in printing performance. Furthermore, the Z calibration of printers tends to drift during repeated use so unlike the other parameters this settings had to be recalibrated multiple times during data collection - each time introducing a little noise. Additionally, the Z probes for measuring the offset add further noise and the bed as it heats and cools in printing cycles can affect the performance. Some very low values in Z offset in the dataset (< 0.08mm) are incorrectly labelled and should be ignored. It was also found that the resolution and quality of some images may affect the performance at detecting small, subtle changes in Z offset.

The data distribution is approximately uniform with some added noise, with the exception of at 0mm where more data was collected. The data predominantly falls in the range of -0.08mm - 0.32mm.

#### Hotend

This is reliable data as it consists of absolute temperature readings from the thermistor mounted to the hotend. All the prints used PLA feedstock and as such these absolute values can be converted to relative values by using a standard sensible PLA temperature range of 190°C - 220°C as a baseline.

The data distribution is approximately Gaussian with some added noise centered at around 205°C. This distribution is because often low temperature prints fully failed or the nozzle was not capable of retaining a constant very high temperature. The data predominantly falls in the range of 160°C - 250°C.

#### Bed

This is reliable data as it consists of absolute temperature readings from the thermistor mounted to the print bed. All the prints used PLA feedstock and as such these absolute values can be converted to relative values by using a standard sensible PLA temperature range of 60°C - 70°C as a baseline. The bed was either set to 60°C or 65°C during printing (the majority of samples are at 65°C).

## Online links

* [https://www.nature.com/articles/s41467-022-31985-y](https://www.nature.com/articles/s41467-022-31985-y) - Link to the Nature Communications Paper.
* [https://doi.org/10.17863/CAM.84082](https://doi.org/10.17863/CAM.84082) - Link to the data repository.
* [https://github.com/cam-cambridge/caxton](https://github.com/cam-cambridge/caxton) - Link to the GitHub repository.

## Authors

* **Douglas A. J. Brion** - *First Author* - [dajb3@cam.ac.uk](mailto:dajb3@cam.ac.uk)
* **Sebastian W. Pattinson** - *Principal Investigator / Supervisor* - [swp29@cam.ac.uk](mailto:swp29@cam.ac.uk)

## License

This project is licensed under the CC BY License - see [About CC Licenses](https://creativecommons.org/about/cclicenses/) for details.

## Acknowledgments

This work has been funded by:

* Engineering and Physical Sciences Research Council, UK Ph.D. Studentship EP/N509620/1 to D.A.J.B.
* Royal Society award RGS/R2/192433 to S.W.P.
* Academy of Medical Sciences award SBF005/1014 to S.W.P.
* Engineering and Physical Sciences Research Council award EP/V062123/1 to S.W.P.
* Isaac Newton Trust award to S.W.P.
