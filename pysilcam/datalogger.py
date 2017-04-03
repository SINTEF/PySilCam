class DataLogger:
    def __init__(self, filename, header):
        self.filename = filename

        #Generate file with header
        with open(filename, 'w') as fh:
            fh.write(header)

    def append_data(self, data):
        #Append data line to file
        with open(self.filename, 'a') as fh:
            #Generate data line string for data list
            line = ','.join([str(el) for el in data])

            #Append line to file
            fh.write(line + '\n')
