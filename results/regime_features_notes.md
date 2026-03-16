# Struggles
## Difference in Data resolution
When deciding on the features being used for the HMM, owing to the fact that most of the data we have is daily, we would be unable to use features such as policy spreads in the model as it would lead to misinterpretation by the model. This is because policy decisions only update every quarter.

Trying to force them into our feature set proved to be a wasted effort, as efforts to difference the spreads, find cycles for the spreads ultimately led to non-persistent regimes that averaged 1-4 days.

**Initial Feature Set**
1. `move_chg` to represent rate volatility
2. `vix_chg` to represent equity volatility
3. `dxy_chg` to represent USD funding stress (FX risk)
4. `cesi_zscore` from US and EUR to represent macro surprise
5. `policy_fed_spread` between Fed-ECB and Fed-BoJ

From the initial feature set, the initial worry was about the stationarity of the policy spreads and how that may affect the HMM. They were trending as opposed to the remaining features that were I(0) stationary. Hence the spreads were differenced.

However upon further testing it was realised that stationarity of the spreads were a red herring. The reason for the weird regime persistence in the HMM is due to the fact that Fed policy decisions were only updated every quarter as opposed to the remaining data being updated daily. Therefore the decision was made to drop the policy spread data as it may negatively affect the HMM.

**Subsequent Trial and Error**
Expanded the feature set to 11 features, including original factors and the 2year-10year yield spreads for each of the countries we are trading (US-EUR, AUD-JPY), along with the currency overnight FX implied volatility to represent foreign exchange volatilities.

The initial exploratory testing of the HMM over the entire 11 features proved to be too volatile due to highly correlated features. This led to too many feature inputs in the HMM which caused low regime persistence and overall sub-optimal results. There were simply too much noise from the feature set and we determined to narrow it down to around 4-5 features that had low correlations to each other.

**Final Feature Set**
We decided to base the feature set based on a couple of high level macro ideas. Which were:
1. Volatility, 
2. USD funding risk 
3. Yield curve signals 
4. Macro shocks
- For volatility measure, we decided to go with the MOVE index changes over the VIX changes as we would be dealing with rate spreads in our strategy and felt that basing the regimes of the MOVE index would be more relevant in our decision making when trading in the different regimes.
- For the USD funding risk measure, we decided to drop all the overnight implied volatility in favour of the `BBDXY` index which tracks the US dollar against 10 global currencies. A drop in the index represents a reduced demand/confidence in the US dollar, and a rise, the inverse.
- For the yield curve signals, we decided to get the PC1 component of the combination of the 2 year-10 year spreads for USD, EUR, AUD, and JPY and represent that as the global yield curve signal
- For the macro shocks, we decided to get the PC1 component of the combination of the CESI for US, EUR, JP, AU

We felt that narrowing it down to 4 features based on overarching macro ideas with relatively low correlation would improve our regime persistence and average regime duration. This would ultimately allow for an effective HMM model that would allow make meaningful use of the information provided to trade on.

Another modification to the feature set was adding a scaling component to the standard deviations based on perceived importance of the feature in determining the regimes. Higher weightage was given to MOVE change and lower was given to funding stress. Yield curve and macro surprises were in the middle.


