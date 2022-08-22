# Dataset viewer app

This a a PoC developed on the road to have a look at the operations done by the `preprocessing/training/01a_catalogue_cleaning_and_filtering/clean.py` script ran on several datasets.

## Launch the app

You will first need to update the `backend/.streamlit/config.toml` file. The `DATASET_DIR_PATH_BEFORE_CLEAN_SELECT` variable must indicate the path to the folder containing the artifacts created by the `checks-save-path`` argument of clean.py. 

For example you should put `DATASET_DIR_PATH_BEFORE_CLEAN_SELECT = "$HOME/data"` for:
```bash
python clean.py \
    [...] 
    --checks-save-path $HOME/data/clean
```

Then after having installed the requirements you can launch the app, for example with the command:
```bash
streamlit run app.py --server.port 8081 --server.fileWatcherType none
```

## Connect to the PoC app

If you have launched the app on a GCP remote host you will need to create a tunnel to connect. For example for the Bigscience project we used something like:

```bash
gcloud compute <VM-NAME> --project=<PROJECT-NAME> -- -L 8081:localhost:8081
```

Then you can access it on your browser. On the previous example `localhost:8081`