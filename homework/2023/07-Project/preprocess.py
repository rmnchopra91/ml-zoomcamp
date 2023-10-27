def remove_not_required_fields(data):
    if "Id" in data:
        data.pop("Id")

    if "default" in data:
        data.pop("default")
    
    # Here we will check the number of fields in request data
    is_data_valid = input_field_validation(data)
    
    return is_data_valid, data

def input_field_validation(data):
  print("key value pair in request data")
  if len(data) != 15 :
    return False
  return True

def test():
  return "this is test function"