import pandas as pd


def filter_table(df: pd.DataFrame, options: str or list = 'all', column: str = None):
    if options == 'all':
        return df

    assert column is not None, "Cannot filter data without column name"

    if isinstance(options, str):
        options = [options]
    return df[df[column].isin(options)]


def select_character(df: pd.DataFrame, character: str):
    return df[df['character'] == character].drop(columns=['character'])


def select_charactevent(df: pd.DataFrame, character: str):
    df = df[df['subjects'] == character].drop(columns=['subjects','event_type']).reset_index(drop=True)
    df = pd.concat([pd.DataFrame(columns=['order'], data=list(range(len(df)))), df], axis=1)
    return df


def select_character_and_events(chars_df: pd.DataFrame, events_df: pd.DataFrame, character: str):
    char_df = select_character(chars_df, character)
    chev_df = select_charactevent(events_df, character)
    return char_df, chev_df


def human_feedback(df_1: pd.DataFrame = None, 
                   df_2: pd.DataFrame = None, 
                   df_3: pd.DataFrame = None, 
                   df_4: pd.DataFrame = None, 
                   df_5: pd.DataFrame = None, 
                  approve_column: str = 'approval', 
                    order_column: str = 'order'):
    
    dfs_out = []
    approvalues = [1, 'true', 'True', 'T', 'Yes', True]

    def feedback(df):
        df = df[df[approve_column].isin(approvalues)]
        if order_column in df.columns:
            df = df.sort_values(by=[order_column], ascending=True)
        return df

    if isinstance(df_1, pd.DataFrame):
        df_1 = feedback(df_1)
        dfs_out.append(df_1)

    if isinstance(df_2, pd.DataFrame):
        df_2 = feedback(df_2)
        dfs_out.append(df_2)

    if isinstance(df_3, pd.DataFrame):
        df_3 = feedback(df_3)
        dfs_out.append(df_3)

    if isinstance(df_4, pd.DataFrame):
        df_4 = feedback(df_4)
        dfs_out.append(df_4)

    if isinstance(df_5, pd.DataFrame):
        df_5 = feedback(df_5)
        dfs_out.append(df_5)

    return dfs_out



