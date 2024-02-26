1. ```teleop.py``` allows you to collect demonstrations.
    - by default each demo is saved as a _.pkl_ file in the _demos_ folder.
    - alternatively, you change the ```storage_path``` parameter while creating the DataLogger object to change the destination of the demos.
    - use the ```combine_all_trajs.py``` script to combine all the individual demos into a single _.pkl_ file for training.
2.  Use ```eval_loop.py``` as a starter to create your own policy inference scripts.
3.  ```get_obs_videos.py``` allows to visualize the demonstrations.
