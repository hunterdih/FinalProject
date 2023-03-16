import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset_path = r'C:\Users\David Hunter\OneDrive\Northeastern Classes\Graduate\EECE5644MachineLearning\Project\archive\DataClean-fullage.csv'
    outfile_path = r'C:\Users\David Hunter\OneDrive\Northeastern Classes\Graduate\EECE5644MachineLearning\Project\archive\CAD_Dataset_Cleaned.csv'
    cad_df = pd.read_csv(dataset_path, delimiter=',')
    with open('DatasetReadme.txt', 'w') as f:
        for col in cad_df.columns:
            counter = 0
            counter_list = []
            unique_types = cad_df[col].unique()
            print(f'{col} {unique_types=}')
            datatype = unique_types.dtype
            if not datatype == 'int64' and not datatype == 'float64':
                for item in unique_types:
                    cad_df = cad_df.replace(to_replace=item, value=counter)
                    counter_list.append(counter)
                    counter += 1
                f.writelines(f'Column: {col} Values: {unique_types} Changed to: {counter_list}\n')

        cad_df = cad_df.drop('sno', axis=1)
        f.close()
    cad_df.to_csv(outfile_path, index=False)