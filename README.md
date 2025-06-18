# AStF: Motion Style Tranfer via Adaptive Statistics Fusor
This is the official implementation of "AStF: Motion Style Tranfer via Adaptive Statistics Fusor". We provide several visualized qualitative results below. Our code will be released upon acceptance.  


## Friendly Reminder: Due to the large number of GIF files, some files may take time to load. If a GIF appears static or fails to display, please wait a moment or directly refresh it to fully load. The complete visualization results will appear shortly.


<img src="assets/teaser/teaser.gif" width="600">    

Our AStF effectively transfers style between two motions, while preserving contents.



## Visualization Results in Xia dataset
|                                                 Style Motion                                                 | Childlike Walk <br/>  <img src="assets/1/childlike_01_000.gif" width="250">  |    Netrual Walk<br/> <img src="assets/1/neutral_01_002.gif" width="250">     | Old Punch <br/> <img src="assets/1/old_18_000.gif" width="250"> | Angry Kick <br/> <img src="assets/1/angry_22_000.gif" width="250">    |
|:------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:----------------------------------------------------------------|:----------------------------------------------------------------------|
|       Content Motion <br/>(Depressed Run) <br/> <img src="assets/1/depressed_13_000.gif" width="250">        |  <img src="assets/1/my_depressed_13_000_childlike_01_000.gif" width="250">   | <img src="assets/1/my_depressed_13_000_neutral_01_002.gif" width="250"> |<img src="assets/1/my_depressed_13_000_old_18_000.gif" width="250">  | <img src="assets/1/my_depressed_13_000_angry_22_000.gif" width="250"> |


|                                             Style Motion                                             |       Old Jump  <br/>  <img src="assets/2/old_16_000.gif" width="250">       |        Sexy Run <br/><img src="assets/2/sexy_13_000.gif" width="250">        | Strutting Walk <br/> <img src="assets/2/strutting_01_001.gif" width="250"> | Depressed Kick <br/> <img src="assets/2/depressed_22_000.gif" width="250"> |
|:----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:---------------------------------------------------------------------------|:---------------------------------------------------------------------------|
| Content Motion <br/>(Childlike Walk) <br/> <img src="assets/2/childlike_01_001.gif" width="250"> | <img src="assets/2/my_childlike_01_001_old_16_000.gif" width="250"> | <img src="assets/2/my_childlike_01_001_sexy_13_000.gif" width="250"> | <img src="assets/2/my_childlike_01_001_strutting_01_001.gif" width="250">        | <img src="assets/2/my_childlike_01_001_depressed_22_000.gif" width="250">  |



## Visualization Results in BFA dataset
|                            Style Motion                             |      Neutral <br/>  <img src="assets/3/Neutral_2.gif" width="250">       | Sneaky <br/> <img src="assets/3/Sneaky_3.gif" width="250"> | Zombie <br/> <img src="assets/3/Zombie_3.gif" width="250">           | Proud <br/>  <img src="assets/3/Proud_2.gif" width="250"> |
|:-------------------------------------------------------------------:|:------------------------------------------------------------------------:|:----------------------------------------------------------:|:--------------------------------------------------------------------|:----------------------------------------------------------|
| Content Motion <br/>(Angry) <br/> <img src="assets/3/Angry_3.gif" width="250"> |                    <img src="assets/3/Angry_3_Neutral_2.gif" width="250">                     |   <img src="assets/3/Angry_3_Sneaky_3.gif" width="250">    | <img src="assets/3/Angry_3_Zombie_3.gif" width="250"> | <img src="assets/3/Angry_3_Proud_4.gif" width="250">      |


## Visualization Results for ablation of skewness and kurtosis
|                           |                                Results in Xia dataset                                |                                       Annotation                                        |                         Results in BFA dataset                          |                                                               Annotation                                                               |  
|:-------------------------:|:------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
|      Content Motion       |   Childlike Walk  <br>  <img src="assets/abla1/childlike_01_001.gif" width="250">    | Childlike style with arms swaying side to side is the style we do not desired to retain |     Angry   <br>   <img src="assets/abla3/Angry_3.gif" width="250">     | Angry style exhibit straight back, heavy foot steps, straight legs, and arms hanging rigidly downward, which are not desired to retain |
|       Style Motion        |     Old Jump    <br>        <img src="assets/abla1/old_16_000.gif" width="250">      | Old style with a hunched back, lowered head, and bent knees is what we aim to transfer  |    Sneaky   <br>  <img src="assets/abla3/Sneaky_3.gif" width="250">     |              A sneaky style with a hunched posture, lowered head, quick short foot steps, and forearms extended forward.               |
|     AStF (Full Model)     |       <img src="assets/abla1/my_childlike_01_001_old_16_000.gif" width="250">        |    Full model effectively transferred the old style while retaining the content walk    |        <img src="assets/abla3/Angry_3_Sneaky_3.gif" width="250">        |                              Full model effectively transferred the sneaky style while retaining content                               |  
|       w/o Skewness        |   <img src="assets/abla1/my_childlike_01_001_old_16_000_wo_skew.gif" width="250">    |                                   Without lower head                                    |    <img src="assets/abla3/Angry_3_Sneaky_3_wo_skew.gif" width="250">    |                                                        Without hunched posture                                                         |
|       w/o Kurtosis        |   <img src="assets/abla1/my_childlike_01_001_old_16_000_wo_kurt.gif" width="250">    |     The arms exhibit a childlike style, swaying side to side, which is not desired      |    <img src="assets/abla3/Angry_3_Sneaky_3_wo_kurt.gif" width="250">    |                                                     Without quick short foot steps                                                     |
| w/o Skewness and Kurtosis | <img src="assets/abla1/my_childlike_01_001_old_16_000_wo_skew_kurt.gif" width="250"> |                                  Without hunched back                                   | <img src="assets/abla3/Angry_3_Sneaky_3_wo_skew_kurt.gif" width="250">  |                                      Without quick short foot steps and forearms extended forward                                      |






## Visualization Results for Comparison

|          Model Variant          |                                 Motion in GIF format                                  | Annotation                                                                                                                                            |
|:-------------------------------:|:------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------|
| Content Motion (Childlike Jump) |             <br/> <img src="assets/4/childlike_16_000.gif" width="250">              | Arms-outstretched is the content that we do not desired to retain                                                                                     |
|  Style Motion (Depressed Walk)  |                <img src="assets/4/depressed_01_000.gif" width="250">                 | Head-down, hunched posture, which represents the style of depressed in the style motion, is what we aim to extract and transfer to the content motion |
|           AStF (Ours)           |      <img src="assets/4/my_childlike_16_000_depressed_01_000.gif" width="250">       | our AStF effectively transferred the depressed style while retaining the content jump                                                                 |  
|        MoST (Kim et al.)        |     <img src="assets/4/most_childlike_16_000_depressed_01_000.gif" width="250">      | Lack of depress style expression                                                                                                                      |
|     GenMoStyle (Kim et al.)     |     <img src="assets/4/genmo_childlike_16_000_depressed_01_000.gif" width="250">     | Lack of depress style expression                                                                                                                      |
|   MotionPuzzle (Jang et al.)    | <img src="assets/4/Style_depressed_01_000_Content_childlike_16_000.gif" width="250"> | Lack of content retention                                                                                                                             |
|           Park et al.           |   <img src="assets/4/ref_childlike_jumping_depressed_walking_fs.gif" width="250">    | Awkward Limbs                                                                                                                                         |





