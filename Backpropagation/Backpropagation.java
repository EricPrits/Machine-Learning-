import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

/**
 * This program is an artificial neural network using a backpropagation algorithm
 * to determine the quality of a type of wine given a number of attributes.
 * @author Eric Prits
 *
 */
public class Backpropagation {
	static double[][] inputData;
	static double[][] testData;
	static double[][] trainingData;
	static double[][] inputWeights;
	static double[][] hiddenWeights;
	final static double learningRate = 0.1; 

	
	/**
	 * This function is used to import the data into an array. 
	 */
	public static void getData(){
		try {
			BufferedReader in = new BufferedReader(new FileReader("src/assignment2data.csv"));
			inputData = new double[2512][12];
			String[] titles = in.readLine().split(",");
			for(int i = 0; i < 2512; i++){
				String[] line = in.readLine().split(",");
				for(int j = 0; j < 12; j++){
					if(j==11){
						inputData[i][j] = Double.parseDouble(Character.toString(line[j].charAt(1)));
					}
					else
						inputData[i][j] = Double.parseDouble(line[j]);
				}
			}
			in.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	/**
	 * Function to normalize all attributes to be between 0 an 1
	 */
	public static void normalize(){
		for(int i=0; i<11;i++){
			double max=0, min =100000;
			for(int j=0; j<2512;j++){
				if(inputData[j][i]<min)
					min = inputData[j][i];
				else if (inputData[j][i]>max)
					max = inputData[j][i];
			}
			for(int j=0; j<2512;j++){
				inputData[j][i]= (inputData[j][i]-min)/(max - min);
			}
		}
	}
	
	/**
	 * This function is used to split the input data into test and training set. 
	 */
	public static void splitData(){
		int trainingPercentage = (int)(0.70 * inputData.length);
		double testPercentage = 0.20 * inputData.length;
		trainingData = new double[trainingPercentage][12];
		testData = new double[inputData.length-trainingPercentage][12];
		for(int i =0; i <inputData.length; i++){
			if(i<trainingPercentage)
				trainingData[i] = inputData[i];
			else
				testData[i-trainingPercentage] = inputData[i];
		}
	}
		
	/**
	 * This function generates random weights for starting values
	 */
	public static void randomHiddenWeights(){
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 12; j++) 
				hiddenWeights[i][j] = Math.random();
		}
	}
	
	/**
	 * This function generates random weights for starting values
	 */
	public static void randomInputWeights(){
		for(int i = 0; i < 12; i++){
			for(int j = 0; j < 11; j++) 
				inputWeights[i][j] = Math.random();
		}
	}
	
	/**
	 * This function is where the backpropagation algorithm is implemented. The while loop
	 * continues to iterate until the error is at an acceptable level or the maximum number of trials is exceeded. Inside the for loop, 
	 * every set of inputs is evaluated for every iteration of the while loop. The weights
	 * for the hidden and input layers are updated after every set of inputs.  
	 */
	public static void training(){
		double error = 1;
		int count = 0;
		double[] desiredOutput = null;			//Desired output based on inputs. 
		double[] hiddenOutput = new double[12]; //Output out of hidden layer.
		double[] finalOutput = new double[3];	//Output from final layer. 
		inputWeights = new double[12][11];
		hiddenWeights = new double[3][12];
		randomInputWeights();
		randomHiddenWeights();
		while(error > 0.05 && count <10000){ //Continue to iterate until error is below 0.05, or after 10000 iterations 
			error  = 0;
			for(int i = 0; i < trainingData.length; i ++){	//Loop through all possible inputs. 
				desiredOutput = new double[3];
				calculateHiddenOutput(hiddenOutput, i, trainingData);		
				calculateFinalOutput(finalOutput, hiddenOutput, i);
				calculateDesiredOutput(desiredOutput, i, trainingData);
				updateHiddenWeights(finalOutput, desiredOutput, hiddenOutput);
				updateInputWeights(finalOutput, desiredOutput, hiddenOutput, i);
				error += MSE(desiredOutput, finalOutput);
			}
			count++;
		}
	}
	
	/**
	 * Calculates outputs from hidden layer. 
	 */
	public static void calculateHiddenOutput(double[] output, int num, double[][] data){
		for(int i = 0; i < 12 ; i++){
			double sum = 0;
			for(int j = 0; j < 11; j++)
				sum += inputWeights[i][j] * data[num][j];
			output[i] = 1 / (1 + Math.exp(-sum));
		}		
	}
	
	/**
	 * Calculates final outputs from outer layer. 
	 */
	public static void calculateFinalOutput(double[] output, double[] input, int num){	
		for(int i = 0; i < output.length ; i++){
			double sum = 0;
			for(int j = 0; j < 12; j++){
				if(input[j] > 0.5)
					input[j] = 0.95;
				else if(input[j] < 0.5)
					input[j] = 0.05;
				sum += hiddenWeights[i][j] * input[j];
			}
			output[i] = 1 / (1 + Math.exp(-sum));
		}		
	}
	
	/**
	 * Function to set the desired output based on a set of inputs. The output is
	 * 3 nodes representing 5, 7 or 8. The values used are 0.05 and 0.95, instead of 0 
	 * and 1, since the Sigmoid function approaches 1 at infinity. 
	 * @param res
	 * @param num
	 */
	public static void calculateDesiredOutput(double[] res, int num, double[][] data){
		if(data[num][11] == 5){
			res[0] = 0.95; res[1]= 0.05; res[2] = 0.05;}
		else if(data[num][11] == 7){
			res[0] = 0.05; res[1] = 0.95; res[2] = 0.05;}
		else if(data[num][11] == 8){
			res[0] = 0.05; res[1]= 0.05; res[2] = 0.95;}
	}
	/**
	 * Updates weights going from hidden layer to output layer. 
	 * @param output final outputs
	 * @param desireOutput Desired outputs
	 * @param input Output from hidden layer
	 */
	public static void updateHiddenWeights(double[] output, double[] desireOutput, double[] input){
		for(int i = 0; i < 3; i++){
			if(output[i] > 0.95)
				output[i] = 0.95; //Change to 0.95 and 0.05 since outputs are these values instead of 1 and 0.
			else if(output[i] < 0.05)
				output[i] = 0.05;
			for(int j = 0; j < 12; j++){
				hiddenWeights[i][j] += learningRate * input[j] * (desireOutput[i] - output[i]) * 
						output[i] * (1 - output[i]);
			}
		}
	}
	
	/**
	 * Updates weights going to hidden layer from inputs. 
	 * @param finalOutput Final output
	 * @param desireOutput Desired output
	 * @param hOutput Output from hidden layer
	 * @param num  Used to determine which row of input data to use
	 */
	public static void updateInputWeights(double[] finalOutput, double[] desireOutput, double[] hOutput, int num){
		for(int i = 0; i < 11; i++){
			for(int j = 0; j < 12; j++){
				for(int k = 0; k < 3; k++){
					inputWeights[j][i] += (desireOutput[k] - finalOutput[k]*finalOutput[k])	* finalOutput[k] * finalOutput[k]*(1 - finalOutput[k]*finalOutput[k])* hiddenWeights[k][j] * hOutput[j] * (1 - hOutput[j]);
				}
			}
		}
	}
	
	public static double MSE(double[] desireOutput, double[] finalOutput){ //Calculate error for iteration. 
		double error = 0;
		for(int i = 0; i < 3; i++){	
			error += (desireOutput[i] - finalOutput[i]) * (desireOutput[i] - finalOutput[i]);
		}
		return error;
	}
	
	/**
	 * This function is used to try and predict the wine quality in the test data. 
	 */
	public static void testing(){
		double[] desiredOutput = new double[3];	//Desired output based on inputs. 
		double[] hiddenOutput = new double[12]; //Output out of hidden layer.
		double[] finalOutput = new double[3];   //Final Output
		int correct = 0, wrong = 0;
		for(int i = 0; i < testData.length; i++){
			calculateHiddenOutput(hiddenOutput, i, testData);
			calculateFinalOutput(finalOutput, hiddenOutput, i);
			calculateDesiredOutput(desiredOutput, i, testData);
			if(finalOutput[0] == desiredOutput[0] && finalOutput[1] == desiredOutput[1] && 
					finalOutput[2] == desiredOutput[2])
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct: " +correct);
		System.out.println("Wrong: " +wrong);
		if(wrong != 0)
			System.out.println("Percentage correct: " +correct/wrong+ "%");
		else
			System.out.println("Percentage correct:  100%");
	}
	
	/**
	 * This function writes the real and predicted outputs of the two algorithms
	 * to a text file named "Outputs.txt"
	 * @throws IOException
	 */
	public static void outputFile() throws IOException{
		BufferedWriter out = new BufferedWriter(new FileWriter("Outputs.txt"));
		out.write("Hidden weight vectors: \n");
		for(int i =0;i<hiddenWeights.length;i++){
			out.write("{");	
			for(int j=0;j<hiddenWeights[i].length;j++){
				out.write(hiddenWeights[i][j] + ",");	
			}
			out.write("}\n");	
		}
		out.write("Input weight vectors: \n");
		for(int i =0;i<inputWeights.length;i++){
			out.write("{");	
			for(int j=0;j<inputWeights[i].length;j++){
				out.write(inputWeights[i][j] + ",");	
			}
			out.write("}\n");	
		}
		out.write("\n\nTesting for algrithom: \n");
		
		out.close();
	}
	
	public static void main(String[] args){
		getData();
		normalize();
		splitData();
		training();	
		testing();
		try {
			outputFile();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
