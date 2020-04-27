# Python3 program to find whether a number 
# is power of 2 or not 
  
# Function to check whether a 
# number is power of 2 or not 
def ispowerof2(num): 
    print num & (num - 1);
    if((num & (num - 1)) == 0): 
        return 1
    return 0
  
# Driver function 
if __name__=='__main__': 
    num = 65536
    print(ispowerof2(num)) 
      
# This code is contributed by 
# Sanjit_Prasad 