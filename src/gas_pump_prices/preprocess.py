
def preprocess(path, var_date): 
    import pandas as pd 

    df_time = pd.read_csv(path)

    df_time[var_date] = pd.to_datetime(df_time[var_date], format="%Y-%m-%d")
    # create month and year columns 
    df_time['month_txt'] = pd.to_datetime(df_time[var_date]).dt.strftime('%b')

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Convert to ordered categorical
    df_time['month_txt'] = pd.Categorical(df_time['month_txt'], 
                                                categories=month_order, 
                                                ordered=True)

    df_time["year"] = pd.to_datetime(df_time[var_date]).dt.year.astype(int)
    df_time['year_txt'] = df_time['year'].astype(str)

    return df_time