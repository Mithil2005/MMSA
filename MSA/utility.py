import h5py
import numpy as np

def inspect_dataset(dataset):
    """
    Inspect the structure and content of a given dataset.

    Parameters:
        dataset (dict): The dataset to inspect, organized as nested dictionaries.

    Outputs:
        Prints detailed information about the keys, data structure, and example content of the dataset.
    """
    # Get all top-level keys in the dataset
    dataset_keys = list(dataset.keys())
    print(f"Top-Level Keys in Dataset: {dataset_keys}\n")

    # Loop through each top-level key (field) in the dataset
    for field in dataset_keys:
        # Number of keys within the current field
        n_keys = len(list(dataset[field].keys()))
        # Get the first key within the current field
        some_id = list(dataset[field].keys())[0]
        # Extract the data point associated with the first key
        data_point = dataset[field][some_id]

        # Print information about the current field
        print(f"Field: {field}")
        print(f"Number of Keys in Field: {n_keys}")
        print(f"Example Key: {some_id}\n")

        # Print detailed information about the example key's structure
        print(f"Subfields in '{field} -> {some_id}': {list(data_point.keys())}")
        print(f"Interval Shape: {list(data_point['intervals'].shape)}")
        print(f"Feature Data Shape: {list(data_point['features'].shape)}")

        print(f"Interval Example: {data_point['intervals'][0]}")
        print("-" * 50)  # Separator for better readability


def inspect_csd_file(file_path):
    """
    Inspect the structure and contents of a `.csd` file using HDF5 format.
    
    Parameters:
        file_path (str): The path to the `.csd` file.
    
    Outputs:
        Prints a detailed hierarchical view of the file's groups, datasets, their shapes, 
        data types, and example values.
    """
    def explore_group(group, level=0, indent=0):
        """
        Recursively explore an HDF5 group and print its structure and contents.
        
        Parameters:
            group (h5py.Group): The current group to explore.
            level (int): The current level in the hierarchy.
            indent (int): Indentation for nested levels to improve readability.
        """
        # Print details about the current group
        print(f"\n{' ' * indent}\033[1;37;44mLevel {level}\033[0m: Group \033[33m{group.name}\033[0m")
        print(f"{' ' * (indent + 2)}Keys present: {list(group.keys())}")

        # Iterate through the group's keys
        for key in group.keys():
            item = group[key]

            # If the item is a group, recursively explore it
            if isinstance(item, h5py.Group):
                explore_group(item, level + 1, indent + 2)
            
            # If the item is a dataset, print its details
            elif isinstance(item, h5py.Dataset):
                print(
                    f"{' ' * (indent + 2)}\033[32mDataset\033[0m: {key}, "
                    f"Shape: {item.shape}, Dtype: {item.dtype}, Example: {item[0] if item.size > 0 else 'Empty Dataset'}"
                )

    # Open the file and start the exploration from the root
    print(f"Inspecting file: {file_path}")
    with h5py.File(file_path, 'r') as file:
        explore_group(file)


def read_csd_file(file_path):
    """
    Reads a .csd file and extracts features and timestamps for each segment.

    Parameters:
        file_path (str): Path to the .csd file.

    Returns:
        dict: A dictionary where keys are segment IDs and values are dictionaries
              containing 'features' and sorted 'timestamps'.
    """
    all_features = {}

    with h5py.File(file_path, 'r') as f:
        # Extract model and data group
        model = list(f.keys())[0]
        data_group = f[model]['data']

        print(f"\033[96m[INFO]\033[0m Processing model: \033[93m{model}\033[0m")

        for segment_id in data_group.keys():
            # Extract features and timestamps for the segment
            features = data_group[segment_id]['features'][:]
            timestamps = data_group[segment_id]['intervals'][:]
            
            # Sort timestamps by the starting time
            sorted_timestamps = timestamps[np.argsort(timestamps[:, 0])]

            # Log segment information
            print(
                f"\033[92m[SEGMENT]\033[0m ID: {segment_id} | "
                f"Features Shape: {features.shape}, Timestamps Shape: {timestamps.shape}"
            )

            # Store features and sorted timestamps in dictionary
            all_features[segment_id] = {
                'features': features,
                'timestamps': sorted_timestamps
            }

    print(f"\033[96m[INFO]\033[0m Successfully read data from file: \033[93m{file_path}\033[0m")
    return all_features
