import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from data.prep_data import load_data
from utils.option_models.derive_ad_price import (
    OptionModel, calibrate_model_for_date, get_rnds, 
    check_rnd_properties, check_single_pdf, MODELS, compute_arrow_debreu_prices
)
from utils.recovery.genralized_recovery import compute_physical_dist

# hyperparamaters:
TOP_N_VOLUME = 50

def calibrate_models(filename):
    df = load_data("data/raw_options_data.zip", "data/DGS3MO.csv")

    # Calibrate across all dates using OptionModel class
    unique_dates = sorted(df['date'].unique())
    print(f"Total dates to calibrate: {len(unique_dates)}")

    calibration_results = {}
    for model_name, initial_params in MODELS.items():
        print(f"Model: {model_name}\n")
        results_for_date = {}
        calibrated_params_accum = []
        
        for idx, date in enumerate(unique_dates[:5]):  # Limiting to first 5 dates
            date_df = df[df['date'] == date].copy()
            # date_df['T'] = (date_df['expiration'] - date_df['date']).dt.days / 365.25
            
            S0 = date_df['Adjusted close'].iloc[0]
            rate = date_df['rate'].iloc[0]
            
            # print(f"[{idx+1}/{min(5, len(unique_dates))}] Calibrating {date.date()} | S0={S0:.2f}, r={rate:.4f}, # options={len(date_df)}")
            
            # Create OptionModel with initial parameters
            option_model = OptionModel(
                model_name=model_name,
                initial_params=initial_params,
                S0=S0,
                rate=rate
            )
            
            # Calibrate
            calibrated_params, rmse, pricer, surface = calibrate_model_for_date(
                date_df, option_model, top_n_volume=TOP_N_VOLUME
                )
            
            result = {
                'date': date,
                'num_options': len(date_df),
                'S0': S0,
                'rate': rate,
                'rmse': rmse,
            }
            if calibrated_params is not None:
                failed_msg = 'failed calibration'
                result['rmse'] = rmse if rmse is not None else failed_msg
                for k in initial_params.keys():
                    result[k] = calibrated_params.get(k, failed_msg)
                calibrated_params_accum.append(result)
            
            results_for_date[date] = {
                'calibrated_model': pricer,
                'market_surface_used': surface,
            }
        
        calibrated_params_df = pd.DataFrame(calibrated_params_accum)
        
        calibration_result = {
            'results_by_date': results_for_date,
            'calibrated_params': pd.DataFrame(calibrated_params_df)
        }
        calibration_results[model_name] = calibration_result
        print("\n" + "="*70)
        print("Calibration Results Summary")
        print("="*70)
        print(calibrated_params_df.head())
        
        with open(f'results/{model_name}-{filename}-calibration.pkl', 'wb') as f:
            pickle.dump(calibration_result, f)
    
    print("\nOverall Summary Statistics:")
    for model_name in calibration_results.keys():
        temp_df = calibration_results[model_name]['calibrated_params']
        param_cols = [
            col for col in temp_df.columns 
            if col not in ['date', 'num_options', 'S0', 'rate', 'rmse']
            ]
        print(f"Model: {model_name}")
        print(temp_df.describe()[param_cols].T)

    return calibration_results


def derive_rnds(calibration_results, filename=''):
    rnds_by_model = {}
    for model_name, model_data in calibration_results.items():
        print(f"Producing RNDs for model: {model_name}")
        rnds_for_dates = {}
        results_by_date = model_data['results_by_date']
        for idx, (ttm, result) in enumerate(results_by_date.items()):
            
            pricer = result['calibrated_model']
            market_surface = result['market_surface_used']
            
            pdfs, cdfs, pdfs_discrete = get_rnds(pricer, market_surface)
            if idx == 0:
                # preview only the first time step
                # can exclude if not in jupyter notebook
                monthly_ttms = [
                    ttm for (i, ttm) in enumerate(market_surface.ttms) 
                    if i in [0, 4, 9, 15, 22]
                    ]
                spot = pricer._model.spot()
                check_rnd_properties(pdfs_discrete, cdfs, monthly_ttms, model_name, S0=spot)
                check_single_pdf(monthly_ttms, pdfs_discrete, S0=spot)
            
            ad_prices = compute_arrow_debreu_prices(pricer, pdfs)
            ad_prices_discrete_pdf = compute_arrow_debreu_prices(pricer, pdfs_discrete)

            rnds_for_dates[ttm] = {
                'ad_prices': ad_prices,
                'pdfs': pdfs,
                'cdfs': cdfs,
                # can replace instantaneous pdfs and ad-prices if better
                'ad_prices_discrete': ad_prices_discrete_pdf,
                'pdfs_discrete': pdfs_discrete,
            }
        
        rnds_by_model[model_name] = rnds_for_dates
        with open(f'results/{model_name}-{filename}-rnds.pkl', 'wb') as f:
            pickle.dump(rnds_for_dates, f)
    
    return rnds_by_model


def recover_fdist(rnds_by_model, calibration_results, filename='', N=12, has_pricer=True):
    piece_wise_cols = N
    model_f_dists = {}
    for model_name, rnds_for_dates in rnds_by_model.items():
        f_dist_by_date = {}
        print(model_name)
        for idx, (dt, rnds) in enumerate(rnds_for_dates.items()):

            pdfs = rnds['pdfs_discrete']
            adp = rnds['ad_prices']
            if has_pricer:
                pricer = calibration_results[model_name]['results_by_date'][dt]['calibrated_model']
                S0 = pricer._model.spot()
            else:
                # choose the first ttm because it's the same across ttms
                ttm_days = list(rnds_for_dates[dt]['pdfs_discrete'].keys())[0]
                S0 = rnds_for_dates[dt]['pdfs_discrete'][ttm_days]['adj_close']
            

            f_dist_pw_ridge, D_inv_pw_ridge, H_inv_pw_ridge = compute_physical_dist(adp, key='pw_ridge', N=piece_wise_cols)
            f_dist_pw_ridge_df = pd.DataFrame(f_dist_pw_ridge, columns=adp.T.columns, index=adp.T.index).T

            f_dist_pw, D_inv_pw, H_inv_pw = compute_physical_dist(adp, key='pw', N=piece_wise_cols)
            f_dist_pw_df = pd.DataFrame(f_dist_pw, columns=adp.T.columns, index=adp.T.index).T

            f_dist_ridge_h1, D_inv_ridge_h1, H_inv_ridge_h1 = compute_physical_dist(adp, key='ridge_h1')
            f_dist_ridge_h1_df = pd.DataFrame(f_dist_ridge_h1, columns=adp.T.columns, index=adp.T.index).T

            f_dist_ridge, D_inv_ridge, H_inv_ridge = compute_physical_dist(adp, key='ridge')
            f_dist_ridge_df = pd.DataFrame(f_dist_ridge, columns=adp.T.columns, index=adp.T.index).T

            if idx == 0:
                ttms = f_dist_ridge_df.columns
                if has_pricer:
                    fig, axs = plt.subplots(nrows=len(ttms)//3, ncols=3, figsize=(10, 16))
                else:
                    fig, axs = plt.subplots(nrows=1, ncols=len(ttms), figsize=(10, 4))

                print(dt)
                for i, ax in enumerate(axs.flatten()):
                    ttm = ttms[i]
                    ax.plot(pdfs[ttm]['strikes'] / S0, pdfs[ttm]['rnd'], label=f'rnd-{model_name}')
                    ax.plot(f_dist_pw_ridge_df.index / S0, f_dist_pw_ridge_df[ttm], label=f'f_dist_pw_ridge-{model_name}')
                    ax.plot(f_dist_pw_df.index / S0, f_dist_pw_df[ttm], label=f'f_dist_pw-{model_name}')
                    ax.plot(f_dist_ridge_h1_df.index / S0, f_dist_ridge_h1_df[ttm], label=f'f_dist_ridge_h1-{model_name}')
                    ax.plot(f_dist_ridge_df.index / S0, f_dist_ridge_df[ttm], label=f'f_dist_ridge-{model_name}')
                    if has_pricer:
                        ax.set_title(f"ttm - {round(ttm * 365, 0)} days")
                    else:
                        ax.set_title(f"ttm - {ttm} days")
                    ax.grid()
                handles, labels = [], []
                for ax in fig.axes:
                    for handle, label in zip(*ax.get_legend_handles_labels()):
                        if label not in labels: # Avoid duplicate labels in the legend
                            handles.append(handle)
                            labels.append(label)
                fig.legend(handles, labels, loc='right', bbox_to_anchor=(0.5, 0.05))
                # plt.tight_layout()
                plt.tight_layout(rect=[0.1, 0.1, 1, 1])
                # fig.suptitle(f"{model_name}- comparing rnd with f dist for {dt}")
                fig.show()
            
            f_dist_by_date[dt] = {
                'pw_ridge': f_dist_pw_ridge_df,
                'pw': f_dist_pw_df,
                'ridge_h1': f_dist_ridge_h1_df,
                'ridge': f_dist_ridge_df,
            }
        with open(f'results/{model_name}-{filename}-f_dist.pkl', 'wb') as f:
            pickle.dump(f_dist_by_date, f)

        model_f_dists[model_name] = f_dist_by_date
    
    return model_f_dists


def main(name="pipeline_results"):
    calibration_results = calibrate_models()
    with open(f'results/{name}-models.pkl', 'wb') as f:
        pickle.dump(calibration_results, f)

    rnds_by_model = derive_rnds(calibration_results)
    with open(f'results/{name}-rnds.pkl', 'wb') as f:
        pickle.dump(rnds_by_model, f)

    model_f_dists = recover_fdist(rnds_by_model, calibration_results, N=12)
    with open(f'results/{name}-f_dist.pkl', 'wb') as f:
        pickle.dump(rnds_by_model, f)
    
    results = {
        'calibration_results': calibration_results,
        'rnds_by_model': rnds_by_model,
        'model_f_dists': model_f_dists,
    }
    with open(f'results/{name}.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results


if __name__ == "__main__":
    main()