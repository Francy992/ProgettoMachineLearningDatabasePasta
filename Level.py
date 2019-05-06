class Level:
    
    def __init__(self):
        self.path = []
        self.list_dict = {}

    def set_path(self, path):
        self.path.append(path)
    
    def get_path(self):
        return self.path
    
    def get_string_path(self):
        path = "\\"
        for x in self.path:
            path += x + '\\'
        path = path[:len(path)-5]
        return path


    def add_dict_on_list(self, key, value):
        self.list_dict[key] = value

    
    