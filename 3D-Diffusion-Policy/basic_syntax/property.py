# Reference : https://www.daleseo.com/python-property/

class Person:
    def __init__(self, first_name, last_name, age):
        self.first_name = first_name
        self.last_name = last_name
        self._age = age
        
    def get_age(self):
        return self._age
    
    def set_age(self, age):
        if age < 0:
            raise ValueError("Invalid age")
        self._age = age
        
    """
        person = Person("Jone", "Doe", 20)
        person.get_age() >>> 20
        
        person.set_age(-1) >>> ValueError : Invalid age
        person.set_age(person.get_age() + 1)
        person.get_age() >>> 21
    
    """
    
    age = property(get_age, set_age) # property()를 사용하면, getter/setter 메서드가 깔끔하게 호출됨. 
                                     # age 필드명을 이용해서 다시 나이 데이터에 접근할 수 있음. 
    
    """ 
        person = Person("Jone", "Doe", 20)
        person.age >>> 20
        
        person.age = -1 >>> ValueError : Invalid age
        person.age = person.age + 1
        person.age >>> 21
    """
    
    @property
    def age(self):
        return self._age
    
    @age.setter
    def age(self, age):
        if age < 0:
            raise ValueError("Invalid age")
        self._age = age
    
    """ 
        property()와 @property 의 가장 큰 이점은 외부에 티 내지 않고 내부적으로 클래스의 필드 접근 방법을 바꿀 수 있다는 점. 
    """
    