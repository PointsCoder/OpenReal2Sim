### how to use this part as pose optimization ###
Here we use simulator to optimize the object pose. I recommend doing this optimization before running/re-running the motion part. 

Please run `python openreal2sim/simulation/isacclab/calib/usd_conversion.py --key {key} `

and `python openreal2sim/simulation/isacclab/calib/sim_vanilla.py --key {key} `

in the isaaclab docker in a row for one key. This will update the optimized mesh as well as the object bounds.

If the optimization result is very unwell because of collision or other problems, please run  `python openreal2sim/simulation/isacclab/calib/rollback.py --key {key} `

The use of these files is to enable accurate grasp pose annotation. 