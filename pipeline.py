import pandas as pd
import pickle
from data.prep_data import load_data
from utils.option_models.derive_ad_price import (
    OptionModel, calibrate_model_for_date, get_rnds, 
    check_rnd_properties, check_single_pdf, MODELS, compute_arrow_debreu_prices
)

# hyperparamaters:
TOP_N_VOLUME = 50

def calibrate_models():
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
            date_df['T'] = (date_df['expiration'] - date_df['date']).dt.days / 365.0
            
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
        
        calibration_results[model_name] = {
            'results_by_date': results_for_date,
            'calibrated_params': pd.DataFrame(calibrated_params_df)
        }
        print("\n" + "="*70)
        print("Calibration Results Summary")
        print("="*70)
        print(calibrated_params_df.head())
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


def derive_rnds(calibration_results):
    rnds_by_model = {}
    for model_name, model_data in calibration_results.items():
        print(f"Producing RNDs for model: {model_name}")
        rnds_for_dates = {}
        results_by_date = model_data['results_by_date']
        for idx, (ttm, result) in enumerate(results_by_date.items()):
            
            pricer = result['calibrated_model']
            market_surface = result['market_surface_used']
            
            pdfs, cdfs = get_rnds(pricer, market_surface)
            if idx == 0:
                # preview only the first time step
                # can exclude if not in jupyter notebook
                monthly_ttms = [
                    ttm for (i, ttm) in enumerate(market_surface.ttms) 
                    if i in [0, 4, 9, 15, 22]
                    ]
                spot = pricer._model.spot()
                check_rnd_properties(pdfs, cdfs, monthly_ttms, model_name, S0=spot)
                check_single_pdf(market_surface.ttms, pdfs, S0=spot)
            
            # ad_prices = None
            ad_prices = compute_arrow_debreu_prices(pricer, pdfs, market_surface)

            rnds_for_dates[ttm] = {
                'ad_prices': ad_prices,
                'pdfs': pdfs,
                'cdfs': cdfs
            }
        
        rnds_by_model[model_name] = rnds_for_dates
    
    return rnds_by_model


def main(name="pipeline_results"):
    calibration_results = calibrate_models()
    rnds_by_model = derive_rnds(calibration_results)
    results = {
        'calibration_results': calibration_results,
        'rnds_by_model': rnds_by_model
    }
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results


if __name__ == "__main__":
    main()