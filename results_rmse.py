import pandas as pd
import glob
import re


def process_df(df, method, model):
    # Calculate the mean for the filtered DataFrame
    mean_df = df.mean().reset_index()
    mean_df.columns = ["metric", f'{method}{model}']

    return mean_df

# Read each file into a DataFrame
IID_pretrained = False

augmentations = ['', '_automasking', '_pseudo_dsms','_pseudo_dsms_automasking','_pseudo_dsms_automasking_noadjust', '_pseudo_dsms_automasking_sploss']  #['', '_pseudo', '_lightinput', '_pseudo_lightinput'] #['', '_add', '_rem', '_addrem'] #['']
methods = ['IID'] #['monodepth2', 'monovit', 'IID'] #

method_strings = "[!s]*" # *mono* or *IID*
clipped = True
distorted = False
singlescale = False
specscale = False
addedspec = False
data = "hkfull" #c3vd" #"hkfull"

if singlescale: scaling = "/singlescale" 
elif specscale: scaling = "/specular_scaling"
else:           scaling = ""

addsptxt = ""
if addedspec:   addsptxt = "/addedspec"

dist = "dist_" if distorted else ""
dist_pre = "" if distorted else "un"

clip = "notclipped" if not clipped else ""


# # cecum only
# videos_to_include = ["cecum_t1_a",
#                     "cecum_t1_b",
#                     "cecum_t2_a",
#                     "cecum_t2_b",
#                     "cecum_t2_c",
#                     "cecum_t3_a",
#                     "cecum_t4_a",
#                     "cecum_t4_b"]

# # others
# videos_to_include = ["desc_t4_a",
#                     "sigmoid_t1_a",
#                     "sigmoid_t2_a",
#                     "sigmoid_t3_a",
#                     "sigmoid_t3_b",
#                     "trans_t1_a",
#                     "trans_t1_b",
#                     "trans_t2_a",
#                     "trans_t2_b",
#                     "trans_t2_c",
#                     "trans_t3_a",
#                     "trans_t3_b",
#                     "trans_t4_a",
#                     "trans_t4_b"]

# LightDepth videos used
videos_to_include = ["cecum_t1_a", "cecum_t2_a", "cecum_t3_a",
                     "sigmoid_t3_a", "desc_t4_a",
                     "transc_t2_a", "transc_t3_a", "transc_t4_a"
                     ]  # Replace with your actual video names

if IID_pretrained:
    method_ext = ["", "(.*?)results.*.csv"]
else:
    method_ext = [f"/*{data}*/models", f"finetuned_mono_{data}_288(.*?)/models/"]    

direc = f"{dist_pre}disttrain/{dist_pre}dist{addsptxt}{scaling}" #"undisttrain/undist" or "disttrain/dist"
file_paths = sorted(glob.glob(f"/raid/rema/outputs_rebuttal/{direc}/{method_strings}{method_ext[0]}/*19results*{clip}*.csv"))
file_paths.extend(sorted(glob.glob(f"/raid/rema/outputs_rebuttal/{direc}/{method_strings}{method_ext[0]}/*19inpaintedresults*{clip}*.csv")))
pattern = fr"/outputs_rebuttal/{direc}/(.*?)/{dist}{method_ext[1]}"
    
# Initialize an empty DataFrame to store the combined mean results
combined_mean_df_fewvideos = pd.DataFrame()
combined_mean_df_allvideos = pd.DataFrame()

for file_path in file_paths:
    df = pd.read_csv(file_path)
    # Add a column with the method/file name
    method, model = re.findall(pattern, file_path)[0] # Extract method name from file path
    if "inpainted" in file_path:
        model = f"{model}_inpainted"

    # Filter the DataFrame to include only the specified videos
    df_filtered = df[df['video'].isin(videos_to_include)]

    mean_df_fewvideos = process_df(df_filtered, method, model)
    mean_df_allvideos = process_df(df, method, model)
    
    # Merge the mean values with the combined DataFrame
    if combined_mean_df_allvideos.empty:
        combined_mean_df_fewvideos = mean_df_fewvideos
        combined_mean_df_allvideos = mean_df_allvideos
    else:
        combined_mean_df_fewvideos = pd.merge(combined_mean_df_fewvideos, mean_df_fewvideos, on="metric", how="outer")
        combined_mean_df_allvideos = pd.merge(combined_mean_df_allvideos, mean_df_allvideos, on="metric", how="outer")



# save to csv
output = f"/raid/rema/outputs_rebuttal/{direc}"
combined_mean_df_allvideos.to_excel(f'{output}/{clip}{augmentations}{data}results_rmse_allvideos.xlsx', index=False)
combined_mean_df_fewvideos.to_excel(f'{output}/{clip}{augmentations}{data}results_rmse_fewvideos.xlsx', index=False)