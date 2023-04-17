# Analysis of the data

Structure of the files in this directory:
```
ROOT:
├───R148V-arc
│   ├───layers
│   └───thresholds
│   └───analysis.csv
│   └───example.png
```

See the code for this analysis in [this repository](https://github.com/DannyAlas/3d-transporters). It's not currnetly intended for anyone to use as it's being updated frequently and not well documented, but please contact me if you have any questions.

The `layers` directory contains the tiff files of the layers. This includes the original layers data and the data after thresholding, segmentation, and colocalization analysis.

The `thresholds` directory contains a visulazation of the thresholding processes for each layer. For now there is no robust way to automatically determine the thresholds, so I ussually deafult to Otsu's method.

The `analysis.csv` files contains the colocalization analysis data. The columns are: 

| Column Name    | Description |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| label          | the label id of this label in colocalization file |
| mean intensity | Value of the mean intensity for this label in reference to what it is colocalized with.  In practice this the higher this number, the more the two overlap (not normalized to area) |
| area           | The area of the label in micro meters |
| bbox           | The bounding box `(min_row, min_col, max_row, max_col)`. Pixels belonging to the bounding box are in the half-open interval `[min_row; max_row)` and `[min_col; max_col)`. |
| centroid       | The centroid of the label in `(row, col)` coordinates |
| coords         | The coordinates of the label in `(row, col)` coordinates. This is a list of tuples for each pixel in the label. |

The `example.png` files are some simple analysis based on the data. Much more can be done.