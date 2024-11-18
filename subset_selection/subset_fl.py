import submodlib as sb

class FLSubsetCreation():
    def __init__(self) -> None:
        pass

    def create_subset(self, data_sijs, k=0.3):
        """
        Creates a subset based on the FacilityLocation objective.

        Args:
            data_sijs: n x m matrix of similarities between the pool (|n|) and queries (|m|)
            k: percentage of subset
        Returns:
            Indicies that make up the subset.
        """
        # scale matrix
        data_sijs = (data_sijs - data_sijs.min()) / (data_sijs.max() - data_sijs.min())

        n, _ = data_sijs.shape

        # use facility location to find subset
        fl = sb.functions.facilityLocation.FacilityLocationFunction(n, mode='dense', sijs=data_sijs, separate_rep=False)
        subset = fl.maximize(budget=int(k * n), optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        return subset
    
    def create_conditional_gain_subset(self, data_sijs, private_sijs, k=0.3):
        """
        Creates a subset based on the FacilityLocationConditionalGain objective.

        Args:
            data_sijs: n x m matrix of similarities between the pool (|n|) and queries (|m|)
            private_sijs: n x n matrix of similarities within the pool (|n|)
            k: percentage of subset
        Returns:
            Indicies that make up the subset.
        """
        # scale matrix
        data_sijs = (data_sijs - data_sijs.min()) / (data_sijs.max() - data_sijs.min())
        private_sijs = (private_sijs - private_sijs.min()) / (private_sijs.max() - private_sijs.min())

        n, num_privates = private_sijs.shape

        # use facility location to find subset
        fl = sb.functions.facilityLocationConditionalGain.FacilityLocationConditionalGainFunction(n, num_privates, data_sijs=data_sijs, private_sijs=private_sijs)
        subset = fl.maximize(budget=int(k * n), optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        return subset

    def create_mutual_information_subset(self, data_sijs, query_sijs, k=0.3):
        """
        Creates a subset based on the FacilityLocationMututalInformation objective.

        Args:
            data_sijs: n x m matrix of similarities between the pool (|n|) and queries (|m|)
            query_sijs: m x m matrix of similarities within the queries (|m|)
            k: percentage of subset
        Returns:
            Indicies that make up the subset.
        """
        # scale matrix
        data_sijs = (data_sijs - data_sijs.min()) / (data_sijs.max() - data_sijs.min())
        query_sijs = (query_sijs - query_sijs.min()) / (query_sijs.max() - query_sijs.min())

        n, num_privates = query_sijs.shape

        # use facility location to find subset
        fl = sb.functions.facilityLocationMutualInformation.FacilityLocationMutualInformationFunction(n, num_privates, data_sijs=data_sijs, query_sijs=query_sijs)
        subset = fl.maximize(budget=int(k * n), optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        return subset