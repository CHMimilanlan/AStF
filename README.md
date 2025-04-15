# AStF: Motion Style Tranfer via Adaptive Statistics Fusor
This is the official implementation of "AStF: Motion Style Tranfer via Adaptive Statistics Fusor"



<img src="assets/teaser.gif" width="400">    

Our AStF effectively transfers style between two motions, while preserving contents.



## Visualization Results in Xia dataset
|                                                 Style Motion                                                 | Childlike Walk <br/>  <img src="assets/1/childlike_01_000.gif" width="250">  |    Netrual Walk <img src="assets/1/neutral_01_002.gif" width="250">     | Old Punch <br/> <img src="assets/1/old_18_000.gif" width="250"> |
|:------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:----------------------------------------------------------------| 
|       Content Motion <br/>(Depressed Run) <br/> <img src="assets/1/depressed_13_000.gif" width="250">        |  <img src="assets/1/my_depressed_13_000_childlike_01_000.gif" width="250">   | <img src="assets/1/my_depressed_13_000_neutral_01_002.gif" width="250"> |<img src="assets/1/my_depressed_13_000_old_18_000.gif" width="250">          |


|                                             Style Motion                                             |       Old Jump  <br/>  <img src="assets/2/old_16_000.gif" width="250">       |        Sexy Run <img src="assets/2/sexy_13_000.gif" width="250">        | Strutting Walk <br/> <img src="assets/2/strutting_01_001.gif" width="250"> |
|:----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:---------------------------------------------------------------------------| 
| Content Motion <br/>(Childlike Walk) <br/> <img src="assets/2/childlike_01_001.gif" width="250"> | <img src="assets/2/my_childlike_01_001_old_16_000.gif" width="250"> | <img src="assets/2/my_childlike_01_001_sexy_13_000.gif" width="250"> | <img src="assets/2/my_childlike_01_001_strutting_01_001.gif" width="250">        |



## Visualization Results in BFA dataset
|                            Style Motion                             |      Neutral <br/>  <img src="assets/3/Neutral_2.gif" width="250">       |          Sneaky <img src="assets/3/Sneaky_3.gif" width="250">           | Zombie <br/> <img src="assets/3/Zombie_3.gif" width="250">           |
|:-------------------------------------------------------------------:|:------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:--------------------------------------------------------------------| 
| Content Motion <br/>(Angry) <br/> <img src="assets/3/Angry_3.gif" width="250"> |                    <img src="assets/3/Angry_3_Neutral_2.gif" width="250">                     | <img src="assets/3/Angry_3_Sneaky_3.gif" width="250"> | <img src="assets/3/Angry_3_Zombie_3.gif" width="250"> |



## Visualization Results for Comparison

|               Content Motion (Childlike Jump)               |              <br/> <img src="assets/4/childlike_16_000.gif" width="250">              |        
|:-----------------------------------------------------------:|:------------------------------------------------------:|
|               Style Motion (Depressed Walk)                 | <img src="assets/4/depressed_01_000.gif" width="250"> |
|                         AStF (Ours)                         |         <img src="assets/4/my_childlike_16_000_depressed_01_000.gif" width="250">                             |        
| MoST (Kim et al.)  | <img src="assets/4/most_childlike_16_000_depressed_01_000.gif" width="250"> |
| Park et al.  | <img src="assets/4/ref_childlike_jumping_depressed_walking_fs.gif" width="250"> |


