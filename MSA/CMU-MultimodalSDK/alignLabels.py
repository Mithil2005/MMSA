def align_dataset_labels(dataset, labels):
    aligned_data = {}
    fields = list(dataset.keys())
    removed = 0
    
    # Process each field and label
    for field in fields:
        log.status(f"Labelling {field}:")
        aligned_data[field] = {}

        data_keys = list(dataset[field].keys())
        
        for data_key in data_keys:
            data_point = dataset[field][data_key]
            data_point_timestamp = data_point['intervals'][0]
            label_key = data_key.split('[')[0]  # Split label key once outside loop
            
            if label_key not in labels:  # Skip if no label data for this key
                removed += 1
                log.warn(f"Couldn't find label for {data_key}")
                continue
            
            label_data = labels[label_key]
            timestamps = label_data['timestamps']
            features = label_data['features']
            length = len(timestamps)

            # Iterate over all label timestamps and try to align
            found = False
            for i in range(length):
                timestamp = timestamps[i]
                label = features[i]

                pad = 0.05 * (timestamp[1] - timestamp[0])

                # If the data point timestamp is within the label timestamp range (with padding)
                if data_point_timestamp[0] + pad >= timestamp[0] and data_point_timestamp[1] - pad <= timestamp[1]:
                    aligned_data[field][data_key] = {
                        'interval': data_point_timestamp,
                        'feature': data_point['features'][0],
                        'label': label
                    }
                    found = True
                    break  # Exit the loop once label is found for the data_point
            
            if(found==False):
                removed += 1
                log.warn(f"Couldn't find label for {data_key}")

        log.success(f"Labeled {field}: successfully")
        log.warn(f"Removed {removed} DataPoints as their labels weren't present")
    return aligned_data

# Example usage
# aligned_data = align_dataset_labels(dataset, labels)
