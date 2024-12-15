import pandera as pa
import pandas as pd


def run_data_validation(data):
    """
    Validates the input car data in the form of a pandas DataFrame against a predefined schema,
    and returns the validated DataFrame.

    This function checks that the columns in the input DataFrame conform to the expected types and value ranges.
    It also ensures there are no duplicate rows and no entirely empty rows.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing car evaluation data. The data is validated based on specific criteria for each column.

    Returns
    -------
    The validated DataFrame that conforms to the specified schema.

    Raises
    ------
    pandera.errors.SchemaError
        If the DataFrame does not conform to the specified schema (e.g., incorrect data types, out-of-range values,
        duplicate rows, or empty rows).
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")    
    if data.empty:
        raise ValueError("Dataframe must contain observations.")

    # Validate data schema with Pandera
    # Correct data types in each column
    # No duplicate observations,
    # No outlier or anomalous values, since all of our data are categorical features, no need for this
    schema = pa.DataFrameSchema(
        {
            'buying': pa.Column(str, pa.Check.isin(['low', 'med', 'high', 'vhigh']), nullable=False),
            'maint': pa.Column(str, pa.Check.isin(['low', 'med', 'high', 'vhigh']), nullable=False),
            'doors': pa.Column(str, pa.Check.isin(['2', '3', '4', '5more']), nullable=False),
            'persons': pa.Column(str, pa.Check.isin(['2', '4', 'more']), nullable=False),
            'lug_boot': pa.Column(str, pa.Check.isin(['small', 'med', 'big']), nullable=False),
            'safety': pa.Column(str, pa.Check.isin(['low', 'med', 'high']), nullable=False),
            'class': pa.Column(str, pa.Check.isin(['unacc', 'acc', 'vgood', 'good']), nullable=False)
        },
        checks=[
            pa.Check(lambda data: ~data.duplicated().any(), error='Duplicate rows found.')
        ]
    )
    schema.validate(data, lazy=True)