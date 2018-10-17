import java.io.IOException;

/**
 * This code is simply used to generate a PerceptronFL object, and to run its functions to train and test the perceptron.
 * @author ericprits
 *
 */
public class Main {

	public static void main(String[] args) {
		try {
			Perceptron percep = new Perceptron();
			percep.trainPerceptronSetosa();				//Train perceptron algorithm
			percep.testPerceptron();					//Test with set weights values
			percep.outputFile();						//Output info to text file
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}	
}
