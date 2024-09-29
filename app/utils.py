import pandas as pd


def debug_llm(prompt: str = None, response: str = None):
    
    print('\n'*3)
    print(prompt)
    print('-'*11)
    print(response)


def filter_table(df: pd.DataFrame, options: str or list = 'all', column: str = None):
    if options == 'all':
        return df

    assert column is not None, "Cannot filter data without column name"

    if isinstance(options, str):
        options = [options]
    return df[df[column].isin(options)]


def human_feedback(df_1: pd.DataFrame = None, 
                   df_2: pd.DataFrame = None, 
                   df_3: pd.DataFrame = None, 
                   df_4: pd.DataFrame = None, 
                   df_5: pd.DataFrame = None, feedback_column: str = 'feedback'):
    
    dfs_out = []

    approvalues = [1, 'true', 'True', 'T', 'Yes', True]

    if isinstance(df_1, pd.DataFrame):
        df_1 = df_1[df_1[feedback_column].isin(approvalues)]
        dfs_out.append(df_1)

    if isinstance(df_2, pd.DataFrame):
        df_2 = df_2[df_2[feedback_column].isin(approvalues)]
        dfs_out.append(df_2)

    if isinstance(df_3, pd.DataFrame):
        df_3 = df_3[df_3[feedback_column].isin(approvalues)]
        dfs_out.append(df_3)

    if isinstance(df_4, pd.DataFrame):
        df_4 = df_4[df_4[feedback_column].isin(approvalues)]
        dfs_out.append(df_4)

    if isinstance(df_5, pd.DataFrame):
        df_5 = df_5[df_5[feedback_column].isin(approvalues)]
        dfs_out.append(df_5)

    return dfs_out
