

def network(images, model):
    crop_shape = (81, 81)
    predictions = model.predict(images.reshape([-1] + list(crop_shape) + [3]))

    return predictions[0][1] > .97
