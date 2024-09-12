#imports
import submodlib as sb
import pickle
import argparse

class FLSubsetCreation():
    def __init__(self) -> None:
        pass

    def create_subset(self, data_sijs, k=0.3):
        # scale matrix
        data_sijs = (data_sijs - data_sijs.min()) / (data_sijs.max() - data_sijs.min())

        n, _ = data_sijs.shape

        # use facility location to find subset
        fl = sb.functions.facilityLocation.FacilityLocationFunction(n, mode='dense', sijs=data_sijs)
        subset = fl.maximize(budget=int(k * n), optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=True, verbose=True)
        
        return subset
    
    def create_conditional_gain_subset(self, data_sijs, private_sijs, k=0.3):
        # scale matrix
        data_sijs = (data_sijs - data_sijs.min()) / (data_sijs.max() - data_sijs.min())
        private_sijs = (private_sijs - private_sijs.min()) / (private_sijs.max() - private_sijs.min())

        n, num_privates = private_sijs.shape

        # use facility location to find subset
        fl = sb.functions.facilityLocationConditionalGain.FacilityLocationConditionalGainFunction(n, num_privates, data_sijs=data_sijs, private_sijs=private_sijs)
        subset = fl.maximize(budget=int(k * n), optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=True, verbose=True)
        
        return subset

    def create_mutual_information_subset(self, data_sijs, query_sijs, k=0.3):
        # scale matrix
        data_sijs = (data_sijs - data_sijs.min()) / (data_sijs.max() - data_sijs.min())
        query_sijs = (query_sijs - query_sijs.min()) / (query_sijs.max() - query_sijs.min())

        n, num_privates = query_sijs.shape

        # use facility location to find subset
        fl = sb.functions.facilityLocationMutualInformation.FacilityLocationMutualInformationFunction(n, num_privates, data_sijs=data_sijs, query_sijs=query_sijs)
        subset = fl.maximize(budget=int(k * n), optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=True, verbose=True)
        
        return subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--util_file', default='mod_dep', type=str)
    args = parser.parse_args()
    subset_creation = FLSubsetCreation()
    subset_creation.create_subset(args.util_file)