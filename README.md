# Percetual_distortion

## Overview

In this research project, our aim is to understand and address the disparity between crowd-sourced annotations for landscape beauty in ground-level images and the actual landscape-related concepts. We extend this inquiry to encompass land-cover classification as another geographical task. Our hypothesis posits that these annotations are only partially related to the landscape, being influenced by various image-specific attributes such as transient weather conditions, artistic choices made by photographers, or objects within the landscape scene.

We further assume that landscape-related concepts exhibit a higher degree of spatial autocorrelation compared to other image-specific attributes. To explore these relationships, we employ a zero-shot vision-language model, CLIP by OpenAI, to extract information on 300 image-specific attributes (refer to `clip_prompts.py` and `clip_processing.py`). We complement this analysis with Local Indicators of Spatial Autocorrelation (LISA) implemented in `LISA_processing.py`. Finally, we evaluate our results using linear probing (`probe_processing.py`).

## Project Structure

- `clip_prompts.py` and `clip_processing.py`: Scripts for extracting information on 300 image-specific attributes using the CLIP model.
- `LISA_processing.py`: Script for conducting Local Indicators of Spatial Autocorrelation (LISA) analysis.
- `probe_processing.py`: Script for linear probing to evaluate the results.
- `SoN_class.py` : Classes that support ScenicOrNot data retrieval and processing for CLIP

## Requirements

Make sure to create a sufficient environment with the necessary packages by using the `requirements.txt` file. If you use conda to manage your environment, execute the following command:

```bash
conda create --name <env_name> --file requirements.txt
