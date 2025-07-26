# Data Dictionary for Credit Risk Dataset

This dictionary describes each column in the dataset, specifying whether it is categorical, ordinal, or numerical.

| Column              | Type         | Description                                                      |
|---------------------|--------------|------------------------------------------------------------------|
| Age                 | Numerical    | Age of the applicant (years)                                     |
| Sex                 | Categorical  | Gender of the applicant (0 = female, 1 = male)                   |
| Job                 | Ordinal      | Job type/level (0 = unskilled, 1 = skilled, 2 = highly skilled, etc.) |
| Housing             | Categorical  | Housing status (0 = own, 1 = rent, 2 = free)                     |
| Saving accounts     | Ordinal      | Savings account level (0 = none, 1 = little, 2 = moderate, 3 = quite rich, 4 = rich) |
| Checking account    | Ordinal      | Checking account level (0 = none, 1 = little, 2 = moderate, 3 = rich) |
| Credit amount       | Numerical    | Amount of credit requested                                       |
| Duration            | Numerical    | Duration of credit (months)                                      |
| Purpose             | Categorical  | Purpose of the loan (0 = car, 1 = furniture, 2 = radio/TV, etc.) |
| target              | Categorical  | Credit risk (0 = good, 1 = bad)                                  |

**Notes:**
- Categorical: No inherent order between values.
- Ordinal: Values have a meaningful order/ranking.
- Numerical: Continuous or discrete numbers.

This dictionary can be updated if new columns or encodings are introduced.
