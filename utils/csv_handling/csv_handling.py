import os
import pandas as pd
from data.prep_data import market_returns
from utils.option_models.derive_ad_price import(
    compute_arrow_debreu_prices,
)

def pdf_csv_to_adp_dicts(suffix):
    mkt = market_returns()
    mkt['date'] = pd.to_datetime(mkt['date'])

    found_files = []
    # for f in found_files:
    #     print(f)
    #     with open(os.path.join('results/', f), 'rb') as file:
    #         calibration_result = pickle.load(file)
    #         print(calibration_result['calibrated_params'][['date', 'rmse']])
    for filename in os.listdir('results/'):
        if filename.endswith(suffix):
            found_files.append(filename)
    pdf_by_models = {}
    for f in found_files:
        print(f)
        with open(os.path.join('results/', f), 'rb') as file:
            pdf = pd.read_csv(file)
            pdf['date'] = pd.to_datetime(pdf['date'])
            pdf = pdf.sort_values(['date', 'strike', 'ttm'])
            pdf_mkt = pdf.merge(mkt, on=['date'], how='left')
            pdfs = {}
            for dt in pdf_mkt['date'].unique():
                date_df = pdf_mkt[pdf_mkt['date'] == dt]
                pdf_ttms = {}
                for T in date_df['ttm'].unique():
                    T_df = date_df[date_df['ttm'] == T]
                    pdf_ttms[T] = {
                        'strikes': T_df['strike'].values,
                        'rnd': T_df['prob'].values,
                        'rf': T_df['rate'].values[0],
                        'adj_close': T_df['adj_close'].values[0],
                    }
                pdfs[dt] = pdf_ttms
        pdf_by_models[f.replace('_risk_neutral_pdf_discrete.csv', '')] = pdfs

    rnds_by_models_discrete = {}
    for model_name, pdfs in pdf_by_models.items():
        print(model_name)
        rnds_for_dates = {}
        for dt, ttm_dict in pdfs.items():
            adp_df = compute_arrow_debreu_prices(
                pricer=None,
                pdfs=ttm_dict,
                rf_in_pricer=False,
            )
            rnds_for_dates[dt] = {
                'ad_prices': adp_df,
                'pdfs_discrete': ttm_dict,
            }
        rnds_by_models_discrete[model_name] = rnds_for_dates
        return rnds_by_models_discrete


def f_dist_to_csv(model_f_dists_discrete, path='results/combined_physical_dists_discrete.csv'):
    dt_dfs = []
    for model_name, f_dist_by_date in model_f_dists_discrete.items():
        print(model_name)
        dts = f_dist_by_date.keys()
        for dt in dts:
            for mthd, df_tmp in f_dist_by_date[dt].items():
                ttm_cols = df_tmp.columns
                df_tmp = df_tmp.reset_index()
                for ttm in ttm_cols:
                    df_ttm = df_tmp[['strike', ttm]]
                    df_ttm['ttm'] = ttm 
                    df = pd.DataFrame({
                        'date': dt,
                        'method': mthd,
                        'strike': df_ttm['strike'],
                        'ttm': ttm,
                        'physical_prob': df_ttm[ttm],
                    })
                    dt_dfs.append(df)

    combined_df = pd.concat(dt_dfs, ignore_index=True)
    combined_df.to_csv(path, index=False)
