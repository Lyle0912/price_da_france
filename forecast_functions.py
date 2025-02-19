import numpy as np
import pandas as pd
from calendar import day_abbr
from sklearn.linear_model import LinearRegression

## Naive Models ##


def forecast_naive_s(Y):
    return Y.values[-1]  # return the last row


## Expert Models ##



def forecast_expert(Y, days, expert_wd=[1, 6, 7], expert_lags=[1, 2, 7]):
    S = Y.shape[1]
    forecast = np.repeat(np.nan, S)

    days_ext = pd.concat((days, pd.Series(days.iloc[-1] + pd.DateOffset(1),
                                          index=[len(days)])))
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday+1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in expert_wd])

    # preparation of lags :
    def get_lagged(lag, Z):
        return np.concatenate((np.repeat(np.nan, lag), Z[:(len(Z) + 1-lag)]))

    coefs = np.empty((S, len(expert_wd)+len(expert_lags)+1))

    for s in range(S):
        # prepare the Y vector
        YREG = Y.iloc[:, s].values

        # get lags
        XLAG = np.transpose([get_lagged(lag=lag, Z=YREG)
                            for lag in expert_lags])

        # combine to X matrix
        XREG = np.column_stack((np.ones(Y.shape[0]+1), WD, XLAG))

        act_index = ~ np.isnan(XREG).any(axis=1)
        act_index[-1] = False  # no NAs and no last obs
        model = LinearRegression(fit_intercept=False).fit(
            X=XREG[act_index], y=YREG[act_index[:-1]])

        forecast[s] = model.coef_ @ XREG[-1]
        coefs[s] = model.coef_

    regressor_names = ["intercept"]+[
        day_abbr[i-1]
        for i in expert_wd]+["lag "+str(lag)
                             for lag in expert_lags]

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}
