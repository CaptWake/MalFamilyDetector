import pathlib
import pandas as pd


from abc import abstractmethod

class FileParser:
    @abstractmethod
    def parse(self, file):
        pass


class FileParserCreator:
    @abstractmethod
    def factory_method(self):
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass

    def create_parser(self) -> FileParser:
        """
        Also note that, despite its name, the Creator's primary responsibility
        is not creating products. Usually, it contains some core business logic
        that relies on Product objects, returned by the factory method.
        Subclasses can indirectly change that business logic by overriding the
        factory method and returning a different type of product from it.
        """

        # Call the factory method to create a dataframe object.
        return self.create_parser()


class CSVcreator(FileParserCreator):
    def create_parser(self) -> FileParser:
        return CSVparser()


class JSONcreator(FileParserCreator):
    def create_parser(self) -> FileParser:
        return JSONparser()

    
class CSVparser(FileParser):
    def parse(self, file):
        return pd.read_csv(file)


class JSONparser(FileParser):
    def parse(self, file):
        return pd.read_json(file)
