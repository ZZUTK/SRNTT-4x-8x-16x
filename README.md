# SRNTT-4x-8x-16x
SRNTT models for 4x, 8x, and 16x. This is simple modification based on [SRNTT](https://github.com/ZZUTK/SRNTT) (4x).
More specifically, 8x and 16x are achieved by adding the subpixel layer(s) before the output layer. 

The models are trained on Painting images ([Google Art](https://commons.wikimedia.org/wiki/Category:Gigapixel_images_from_the_Google_Art_Project)) for 30 epochs (first 10 epochs with L1 loss only and then 20 epochs with all losses).
The other settings are the same as the original SRNTT.

**Note: These models do not ensure the optimal performance.**

## Test

```bash
python eval.py --scale 16
```

`--scale` can be 4, 8, or 16, which indicate the up-scaling factor. The results will be saved to `./result_16x` by default.
