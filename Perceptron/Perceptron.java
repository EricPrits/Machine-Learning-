import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Perceptron {
	private double[] initialw;					//Initial weight values for setosa non-setosa percetron
	private double[] initialw2;					//Initial weight values for second perceptron
	private double[] w;							//Weight values for setosa non-setosa percetron
	private double[] w2;						//Weight values for second perceptron
	private String[] values;					//Holds lines of input data
	private int classificationError;			//Total classification error
	private String[] testValues;				//Holds lines of output data
	private double[] x1 = new double[120];		//First input, Sepal length
	private double[] x2 = new double[120];		//Second input, Sepal width
	private double[] x3 = new double[120];		//Third input, Petal length
	private double[] x4 = new double[120];		//Fourth input, Petal width
	private String[] name = new String[120];	//Name of flower species
	private double[] classified = new double[120];	//Name of flower species
	final double learningRate = 0.1;			
	int theta = 0;
	String[] percepTest;
	String[] errorTest;
	
	/**
	 * This constructor is used to import the training and testing data 
	 * into appropriate arrays.
	 * @throws IOException
	 */
	Perceptron() throws IOException{
		w = new double[5];
		w2 = new double[5];
		initialw = new double[5];
		initialw2 = new double[5];
		values = new String[120];
		BufferedReader in = new BufferedReader(new FileReader("src/train.txt"));
	    for(int i = 0; i < 120; i++){
	    	values[i] = in.readLine(); //inputs every line of data into array
	    }
	    String[] line;
	    for(int i = 0; i < values.length; i++){ //Splits data into 5 separate inputs
	    	line = values[i].split(",");
	    	x1[i] = Double.parseDouble(line[0]); 
	    	x2[i] = Double.parseDouble(line[1]); 
	    	x3[i] = Double.parseDouble(line[2]); 
	    	x4[i] = Double.parseDouble(line[3]); 
	    	name[i] = line[4]; //class
	    }
	    in.close();
	    
	    testValues = new String[30];
	    BufferedReader in2 = new BufferedReader(new FileReader("src/test.txt"));
	    for(int i = 0; i < 30; i++){
	    	testValues[i] = in2.readLine(); //import testing data
	    }
	    in2.close();
	}
	
	/**
	 * This function is used to adjust the weights to learn the separation between
	 * Iris-setosa and not Iris-setosa. A simple feedback learning perceptron algorithm is used to adjust the 
	 * weights. As these are linearly separable the algorithm will be perfect. Once the not-setosas are determined 
	 * a second perceptron is used to see if they versicolor or virginica
	 */
	public void trainPerceptronSetosa(){
		int count = 0, e = 0; //Counts number of iterations
		int desiredOutput, output; 	//Calculated output and expected output
		double lError;	//error values
		randomWeights(); //Set weights to initial random values
		do{ 
			count++;
			e = 0;
			for(int i = 0; i < x1.length; i++){ //Test each data point
				output = calculateOutputSetosa(x1[i], x2[i], x3[i], x4[i]); //determine calculated output
				desiredOutput = calculateDesiredOutputSetosa(name[i]); //determine real output
				lError = desiredOutput - output;
				classificationError += lError *lError;
				if(lError == 0){ //If no error has occurred 
					e++;
					classified[i]=output;
				}
				//update weight values, lError will be 1 or -1 indicating whether to increase or decrease the weights
				w[0] += learningRate * lError * x1[i];
				w[1] += learningRate * lError * x2[i];
				w[2] += learningRate * lError * x3[i];
				w[3] += learningRate * lError * x4[i];
				w[4] += learningRate * lError;				
			}
		}while(x1.length - e != 0); //iterate until no errors
		//Now the the setosa and non-setosas have been split apart call the second perceptron to split the not-setosas
		//Pass count to ensure the final number of iterations is correct
		trainPerceptron(count);
	}
	/**
	 * Splits the non-setosa values into either versicolor or virginica
	 * @param count, the amount of iterations that it took to split into setosa and non-setosa
	 */
	public void trainPerceptron(int count){
		int e; //Counts how many errors have occurred in one given pass
		int desiredOutput, output; 	//Calculated output and expected output
		double lError;	//error values
		randomWeights2(); //Set weights to initial random values
		do{ 
			e = 40;
			count++;
			for(int i = 0; i < x1.length; i++){ //Test each data point
				if(classified[i]==0){//For the non-Setosa flowers as the setosa flowers are already calculated
					output = calculateOutput(x1[i], x2[i], x3[i], x4[i]); //determine calculated output
					desiredOutput = calculateDesiredOutput(name[i]); //determine real output
					lError = desiredOutput - output;//Determine if the output matches the desired output  
					classificationError += lError *lError;
					if(lError == 0) //If no error has occurred 
						e++;
					//update weight values, lError will be 1 or -1 indicating whether to increase or decrease the weights
					w2[0] += learningRate * lError * x1[i];
					w2[1] += learningRate * lError * x2[i];
					w2[2] += learningRate * lError * x3[i];
					w2[3] += learningRate * lError * x4[i];
					w2[4] += learningRate * lError;		
				}
			}	
		}while(x1.length - e != 0); //iterate until no errors
       System.out.println("\nNumber of iterations for perceptron: " +count); 
	}
	
	
	/**
	 * This function is used to predict if a set of inputs belong to Iris-setosa 
	 * or not based on the weights set in the trainPerceptron algorithm.
	 * @throws IOException
	 */
	public void testPerceptron() throws IOException{
		//Get test data 
		double testx1[] = new double[30];
		double testx2[] = new double[30];
		double testx3[] = new double[30];
		double testx4[] = new double[30];
		String[] testName = new String[30];
	    String[] line;
	    //Set arrays for test data points
	    for(int i = 0; i < testValues.length; i++){
	    	line = testValues[i].split(",");
	    	testx1[i] = Double.parseDouble(line[0]);
	    	testx2[i] = Double.parseDouble(line[1]);
	    	testx3[i] = Double.parseDouble(line[2]);
	    	testx4[i] = Double.parseDouble(line[3]);
	    	testName[i] = line[4];
	    }
	    percepTest = new String[30];
	    for(int i = 0; i < testx1.length; i++){ //Test every data set
	    	int output = calculateOutputSetosa(testx1[i], testx2[i], testx3[i], testx4[i]);
	    	if(output == 1) //Output of 1 indicates Setosa
	    		percepTest[i] = "Iris-setosa";
	    	else{//Output of 0 indicates it is not-setosa
	    		output=calculateOutput(testx1[i],testx2[i],testx3[i],testx4[i]);
	    		if(output ==1)
		    		percepTest[i] = "Iris-versicolor";
	    		else
	    			percepTest[i] = "Iris-virginica";
	    	}
	    }
	    
	}	
	/**
	 * 
	 * @return Returns 1 or 0 depending if the sum is greater than theta or not based on the weights for the first perceptron
	 */
	public int calculateOutputSetosa(double x1, double x2, double x3, double x4)
	{
		double sum = x1 * w[0] + x2 * w[1] + x3 * w[2] + x4 * w[3] + w[4];
		return (sum >= theta) ? 1 : 0;
	}
	/**
	 * 
	 * @return Returns 1 or 0 depending if the sum is greater than theta or not based on the weights for the second perceptron
	 */
	public int calculateOutput(double x1, double x2, double x3, double x4)
	{
		double sum = x1 * w2[0] + x2 * w2[1] + x3 * w2[2] + x4 * w2[3] + w2[4];
		return (sum >= theta) ? 1 : 0;
	}
	
	
	/**
	 * 
	 * @param name Name of the flower species
	 * @return Returns 1 if sestosa, 0 otherwise
	 */
	public int calculateDesiredOutputSetosa(String name){  
		if(name.charAt(5) == 's') //For some reason my program was having trouble equating entire word
			return 1;
		else
			return 0;
	}
	
	
	/**
	 * 
	 * @param name Name of flower species
	 * @return 1 if virsicolor, 0 if virginica
	 */
	public int calculateDesiredOutput(String name){
		if(name.charAt(6) == 'e') //For some reason my program was having trouble equating entire word
			return 1;
		else
			return 0;
	}
	
	
	/**
	 * This function generates random weights for the Setosa and Not-Setosa starting values
	 */
	public void randomWeights(){
		for(int i = 0; i < 5; i++){
			w[i] = Math.random(); 
			initialw[i]=w[i];
		}
	}
	/**
	 * This function generates random weights for starting values for the second perceptron
	 */
	public void randomWeights2(){
		for(int i = 0; i < 5; i++){
			w2[i] = Math.random();
			initialw2[i]=w2[i];
		}
	}
	
	
	/**
	 * This function writes the real and predicted outputs of the two algorithms
	 * to a text file named "Outputs.txt"
	 * @throws IOException
	 */
	public void outputFile() throws IOException{
		BufferedWriter out = new BufferedWriter(new FileWriter("Outputs.txt"));
		out.write("Initial and final weight vectors for first perceptron: \n");
		out.write("Initial: " + initialw[0] + "," + initialw[1] + "," + initialw[2] + "," + initialw[3] + "," + initialw[4]);
		out.write("\nFinal: " + w[0] + "," + w[1] + "," + w[2] + "," + w[3] + "," + w[4]);
		out.write("\n\nInitial and final weight vectors for second perceptron: \n");
		out.write("Initial: " + initialw2[0] + "," + initialw2[1] + "," + initialw2[2] + "," + initialw2[3] + "," + initialw2[4]);
		out.write("\nFinal: " + w2[0] + "," + w2[1] + "," + w2[2] + "," + w2[3] + "," + w2[4]);
		out.write("\n\nTotal Classification Error: " + classificationError);
		out.write("\n\nNumber of iterations for perceptron: ");
		out.write("\n\nTesting for perceptron: \n");
		for(int i = 0; i < testValues.length; i++){
			String[] name = testValues[i].split(",");
			out.write("\nReal Value: ");
			out.write(name[4]);
			out.write("\tPredicted Value: ");
			out.write(percepTest[i]);
		}
		out.write("\n\nPrecision of classification:");
		out.write("\n\nRecall of classification:");
		
		out.close();
	}
}
