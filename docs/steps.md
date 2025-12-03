Here is the step-by-step process:
Process to Estimate the Pareto Parameter (q)

The parameter q (or α in some texts) is estimated only from the largest claims, which is the tail of the loss distribution.
Step 1: Data Preparation and Threshold Selection (K)

Before estimation, the data must be clean, merged, and cut at the appropriate threshold.

    Merge Data: Every large claim loss (xi​) in the insurer's data must be successfully linked to the corresponding climate variable reading (e.g., Soil Moisture Index) from the climate dataset using the Date of Loss and Geographic Coordinates.

    Select the Threshold (K): This is the minimum loss size above which the data is assumed to be Pareto-distributed.

        Method: Thabile should use an Actuarial Method like the Mean Excess Plot (MEP) to find the point K where the plot becomes linear. This linearity visually confirms the Pareto (heavy-tail) assumption holds for all losses xi​>K.

    Filter the Data: Filter the merged dataset to include only claims xi​ where xi​>K. Let n be the total count of these large claims.

Step 2: Baseline Estimation (Static q)

Thabile must first calculate the static q^​ as a baseline to benchmark her later climate-adjusted results. This is done using Maximum Likelihood Estimation (MLE).

    Apply the MLE Formula: Use the single-parameter Pareto MLE formula, which she noted in her project update:
    q^​=∑i=1n​ln(Kxi​​)n​

        n: The number of claims greater than the threshold K.

        xi​: The size of the i-th claim (loss amount).

        K: The chosen threshold.

        Interpretation: This q^​ represents the average tail heaviness based on the entire historical period.

Step 3: Non-Stationary Modeling (Dynamic q via Climate)

This is the advanced step of her research: making q a function of a climate variable (e.g., Soil Saturation Index, SSI).

    Identify the Severity Driver: Select the single most important climate variable (e.g., SSI) that is statistically proven to influence the severity (size) of claims.

    Choose the Link Function (GLM): Instead of a single q^​, she needs a relationship that says: "As the climate driver increases, q must decrease (making the tail heavier)." This is typically done using a Generalized Linear Model (GLM) where the log of q is linked to the climate covariate.

        Model Structure:
        ln(q^​t​)=β0​+β1​×SSIt​

        Note: The coefficient β1​ is expected to be negative. If SSI is high (adverse climate), the term β1​×SSIt​ decreases, forcing ln(q^​t​) down, which makes q^​t​ smaller.

    Estimate Coefficients (β): Using the filtered claims data and their linked SSI values, run the GLM estimation to find the numerical values for the intercept (β0​) and the climate factor (β1​).

Step 4: Final q Calculation for Scenarios

Once the β coefficients are estimated, Thabile has a dynamic model for q.

    Calculate Scenario q^​: Use the estimated GLM (from Step 3) and input a specific climate scenario value for SSI to get the adjusted q^​:

        Historical Average: Input the average SSI over the historical period to verify the result is close to the baseline q^​ from Step 2.

        Adverse Scenario: Input a high SSI value (e.g., the 95th percentile SSI from the last 10 years, or a projected value from climate models). This will yield the lower, heavier-tailed q^​Adverse​.

    Integrate into GEMAct: Feed this specific q^​Adverse​ value into the GEMAct package to run the aggregate loss simulation under the Adverse Climate Scenario.

    Measure Impact: Compare the resulting TVaR (Tail Value-at-Risk) from the q^​Adverse​ run to the TVaR from the q^​Base​ run. The difference is the measure of climate risk.
