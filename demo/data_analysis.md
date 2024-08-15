from collections import Counter

import matplotlib.pyplot as plt

# Connect cBioPortal API

`from bravado.client import SwaggerClient`
`cbioportal = SwaggerClient.from_url('https://www.cbioportal.org/api/v2/api-docs', config={"validate_requests": False, "validate_responses": False, "validate_swagger_spec": False})`


This will give a dropdown with all the different APIs, similar to how you can see them here on the cBioPortal website: https://www.cbioportal.org/api/swagger-ui/index.html.

## Endpoints
You can get the name of an endpoint by going to that website and clicking on any of the endpoints. The URL will then update to show the name of the endpoint. For example, the endpoint for getting a cancer type is getCancerTypesUsingGET, which shows up on the website as https://www.cbioportal.org/api/swagger-ui/index.html#/Cancer%20Types/getCancerTypeUsingGET.

The documentation shows one of the parameters is cancerTypeId of type string, the example acc is mentioned:

`acc = cbioportal.Cancer_Types.getCancerTypeUsingGET(cancerTypeId='acc').result()`

### cBioPoral stores cancer genomics data from a large number of published studies
`studies = cbioportal.Studies.getAllStudiesUsingGET().result()`
`cancer_types = cbioportal.Cancer_Types.getAllCancerTypesUsingGET().result()`

Let's see which study has the largest number of samples:
`sorted_studies = sorted(studies, key=lambda x: x.allSampleCount)`
`sorted_studies[-1]`

### get patient data
Example of study with id msk_impact_2017 study with 10,000 patients sequenced.
`patients = cbioportal.Patients.getAllPatientsInStudyUsingGET(studyId='msk_impact_2017').result()`
`print("The msk_impact_2017 study spans {} patients".format(len(patients)))`

### get mutation data
A study can have multiple molecular profiles. This is because samples might have been sequenced using different assays (e.g. targeting a subset of genes or all genes). An example for the acc_tcga study is given for a molecular profile (acc_tcga_mutations) and a collection of samples (msk_impact_2017_all). We can use the same approach for the msk_impact_2017 study. This will take a few seconds. You can use the command %%time to time a cell):
`%%time`
`mutations = cbioportal.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(`
`    molecularProfileId='msk_impact_2017_mutations,`
`    sampleListId='msk_impact_2017_all',`
`    projection='DETAILED'`
`).result()`

We can explore what the mutation data structure looks like:
`mutations[0]`


Now that we have the gene field we can check what gene is most commonly mutated:
`mutation_counts = Counter([m.gene.hugoGeneSymbol for m in mutations])`
`mutation_counts.most_common(5)`


### How many samples have a TP53 mutation?
For this exercise it might be useful to use a pandas dataframe to be able to do grouping operations. You can convert the mutations result to a dataframe like this:

```python
import pandas as pd
mdf = pd.DataFrame.from_dict([
    # python magic that combines two dictionaries:
    dict(
        {k: getattr(m, k) for k in dir(m)},
        **{k: getattr(m.gene, k) for k in dir(m.gene)})
    # create one item in the list for each mutation
    for m in mutations
])
```
Now that you have the data in a Dataframe you can group the mutations by the gene name and count the number of unique samples in TP53:
```python
sample_count_per_gene = mdf.groupby('hugoGeneSymbol')['uniqueSampleKey'].nunique()
print("There are {} samples with a mutation in TP53".format(
    sample_count_per_gene['TP53']
))
```

## Visualization 
It would be nice to visualize this result in context of the other genes by plotting the top 10 most mutated genes. For this you can use the matplotlib interface that integrates with pandas.
`%matplotlib inline`
`sample_count_per_gene.sort_values(ascending=False).head(10).plot(kind='bar') `
`plt.show()`
