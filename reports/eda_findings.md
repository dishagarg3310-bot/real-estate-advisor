# EDA Findings: India Housing Prices

Dataset: `india_housing_prices.csv` (250,000 properties, 23 original columns). All charts are in the `reports/` folder and were produced by the scripts `src/04_eda_price_size.py`, `src/05_eda_location.py`, `src/06_eda_relationships.py` and `src/07_eda_investment.py`.

## Data quality summary

| Check | Result | How it was handled |
|---|---|---|
| Missing values | 0 | Nothing to impute |
| Duplicate rows / duplicate IDs | 0 / 0 | Nothing to remove (duplicate check kept in the cleaning script) |
| Price per sq ft unit | Stored in lakhs and rounded to 2 decimals | Recalculated exactly in rupees (`Price_per_SqFt_Rupees`) |
| Floor_No higher than Total_Floors | 116,304 rows (46.5%) | Kept and flagged with `Floor_Inconsistent`, because deleting them would remove almost half of the data |
| Amenities | Comma-separated text, 325 combinations | Converted to `Amenity_Count` and one 0/1 column for each of the 5 amenities |
| Outliers | None in size or price; 19,723 (7.9%) in price per sq ft | Kept: they are small flats with high prices, not errors |
| Age_of_Property | Calculated from the year 2025 | Used as given |

## Section 1: Price and size analysis (questions 1 to 5)

**1. What is the distribution of property prices?**
Prices range from 10 to 500 lakhs and are spread almost evenly: about 10% of properties fall in each 50-lakh band. The mean (254.6) and median (253.9) are almost equal and the skew is 0.008, so the distribution is symmetric with no dominant price segment.
Chart: `eda_q01_price_distribution.png`

**2. What is the distribution of property sizes?**
Sizes range from 500 to 5,000 sq ft and are also spread evenly (mean 2,750, median 2,747, skew 0.001).
Chart: `eda_q02_size_distribution.png`

**3. How does price per sq ft vary by property type?**
Median price per sq ft is Rs 9,245 for Apartments, Rs 9,305 for Independent Houses and Rs 9,190 for Villas, a spread of only about 1.3%. Property type has almost no effect on price per sq ft. The means (about Rs 13,000) are higher than the medians because of a few very high values.
Chart: `eda_q03_ppsf_by_type.png`

**4. Is there a relationship between property size and price?**
No. The correlation is -0.0025, and the median price is between 252.6 and 255.8 lakhs in every size quintile. A 5,000 sq ft property costs about the same as a 500 sq ft one in this dataset.
Chart: `eda_q04_size_vs_price.png`

**5. Are there any outliers in price per sq ft or property size?**
Size and price have no outliers (0 by the IQR rule). Price per sq ft has 19,723 high outliers (7.9% of properties). These properties average only 841 sq ft compared with 2,750 for all properties: because price and size are unrelated, a small flat with a high price gets an extreme price per sq ft. They were kept.
Chart: `eda_q05_outliers.png`

## Section 2: Location analysis (questions 6 to 10)

**6. What is the average price per sq ft by state?**
The highest are Karnataka (Rs 13,252), Andhra Pradesh (Rs 13,202) and Uttar Pradesh (Rs 13,143); the lowest are Delhi (Rs 12,941), Rajasthan (Rs 12,932) and Punjab (Rs 12,931). The gap between the highest and lowest of the 20 states is only 2.5%.
Chart: `eda_q06_ppsf_by_state.png`

**7. What is the average property price by city?**
The highest are Bangalore (258.5 lakhs), Surat (258.1) and Kochi (257.4); the lowest are New Delhi (251.6), Trivandrum (251.3) and Cuttack (250.8). Across the 42 cities the gap is only 3.1%, so city is not a strong price driver in this data.
Chart: `eda_q07_price_by_city.png`

**8. What is the median age of properties by locality?**
Across 500 localities the median age is between 16 and 21 years, and the overall median is 18. Localities are very similar in age; the oldest are Locality_72 and Locality_403 (21 years).
Chart: `eda_q08_age_by_locality.png`

**9. How is BHK distributed across cities?**
Each BHK type (1 to 5) makes up about 20% of all properties. Inside a single city the share of a BHK type ranges only from 18.6% to 22.0%, so every city has a similar mix.
Chart: `eda_q09_bhk_by_city.png`

**10. What are the price trends for the top 5 most expensive localities?**
The top 5 by average price are Locality_395 (275.4 lakhs), Locality_366 (273.4), Locality_497 (270.0), Locality_360 (269.8) and Locality_203 (269.0). The dataset has no sale dates, so the trend is shown by the period in which the property was built. Prices move up and down between periods without a consistent pattern; across the whole dataset the correlation between year built and price is about 0.003.
Chart: `eda_q10_top5_locality_trend.png`

## Section 3: Feature relationships and correlation (questions 11 to 15)

**11. How are numeric features correlated with each other?**
Only three pairs are correlated. Year_Built and Age_of_Property are perfectly negatively correlated (-1.000), because one is calculated from the other. Size vs price per sq ft (-0.615) and price vs price per sq ft (0.556) are also expected, because price per sq ft is price divided by size. Every other pair has a correlation below 0.01 in absolute value, so the original features are essentially independent of each other.
Chart: `eda_q11_correlation_heatmap.png`

**12. How do nearby schools relate to price per sq ft?**
Not at all: the correlation is -0.0001. The average price per sq ft stays between Rs 12,888 and Rs 13,292 for 1 to 10 nearby schools, with no pattern.
Chart: `eda_q12_schools_vs_ppsf.png`

**13. How do nearby hospitals relate to price per sq ft?**
Also not at all: the correlation is -0.0001, and the average stays between Rs 12,917 and Rs 13,257.
Chart: `eda_q13_hospitals_vs_ppsf.png`

**14. How does price vary by furnished status?**
It does not. Average prices are 254.4 lakhs (Furnished), 254.3 (Semi-furnished) and 255.0 (Unfurnished), and average price per sq ft is about Rs 13,000 to Rs 13,100 for all three.
Chart: `eda_q14_price_by_furnished.png`

**15. How does price per sq ft vary by facing direction?**
Median price per sq ft is Rs 9,234 (East), Rs 9,219 (North), Rs 9,225 (South) and Rs 9,308 (West). West is about 1% higher than the others, which is too small to matter.
Chart: `eda_q15_ppsf_by_facing.png`

## Section 4: Investment, amenities and ownership (questions 16 to 20)

**16. How many properties belong to each owner type?**
Brokers 83,479 (33.4%), Owners 83,268 (33.3%) and Builders 83,253 (33.3%): an equal split.
Chart: `eda_q16_owner_type.png`

**17. How many properties are available under each availability status?**
Under_Construction 125,035 (50.0%) and Ready_to_Move 124,965 (50.0%). The dataset has no "Sold" status.
Chart: `eda_q17_availability.png`

**18. Does parking space affect property price?**
Barely. The average price is 254.4 lakhs without parking and 254.7 with parking (+0.1%), and average price per sq ft is slightly lower with parking (Rs 13,045 vs Rs 13,072).
Chart: `eda_q18_parking.png`

**19. How do amenities affect price per sq ft?**
Very little. The correlation between amenity count and price per sq ft is 0.002, and the average rises only from Rs 13,015 (1 amenity) to Rs 13,102 (5 amenities). Properties with a Gym have the largest premium (about +Rs 98, +0.75%); Playground is slightly negative (-Rs 23).
Chart: `eda_q19_amenities.png`

**20. How does public transport accessibility relate to price per sq ft or investment potential?**
Price per sq ft is slightly lower with better transport (Rs 13,086 for Low, Rs 13,062 for Medium, Rs 13,027 for High), a difference below 0.5%. The share of "Good Investment" properties is 65.3% for High transport but only 39.9% (Low) and 39.7% (Medium). This gap is created by the rule itself, which gives one point for High transport. With that point removed the shares are 39.9%, 39.7% and 40.4%, so transport access has no real effect on investment potential in this data.
Chart: `eda_q20_transport.png`

## Overall conclusion

The dataset is clean but behaves like synthetic data. Prices, sizes, BHK, ages and categories are spread almost uniformly, and none of the original features (size, BHK, city, locality, schools, hospitals, furnishing, facing, parking, amenities, transport) explains price in any meaningful way. Consequences for the modelling stage:

- The two targets (`Future_Price_5Y` and `Good_Investment`) do not exist in the data and are built from transparent rules (see the target definitions in `src/03_create_targets.py`).
- Because the targets come from rules on the same columns, high model scores mainly show that the models learned the rules. This must be stated in the final documentation, and the columns that define the targets are excluded from the model inputs.
- No real-world claim about which cities, amenities or features raise property prices can be made from this data.
