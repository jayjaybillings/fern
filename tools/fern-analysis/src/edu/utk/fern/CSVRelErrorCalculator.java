package edu.utk.fern;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

/**
 * 
 */

/**
 * A simple Java code for computing the relative error between the single
 * precision and double precision results.
 * 
 * The results were stored as "log(t) log(y_1) log(y_2) ... log(y_n)" so this
 * code converts everything to a regular scale and then computes the
 * differences.
 * 
 * @author Jay Jay Billings
 *
 */
public class CSVRelErrorCalculator {

	private static final String token = " ";

	/**
	 * The main method
	 * 
	 * @param args
	 *            the input arguments
	 */
	public static void main(String[] args) {

		// Local Declarations
		double err = 0.0, doubleVal = 0.0, singleVal = 0.0;
		String[] singleLine;
		String[] doubleLine;

		// Get the single precision lines
		File singlePFile = new File("../../data/results/fern_150_single.dat");
		ArrayList<String[]> singleLines = getLines(singlePFile);

		// Get the double precision lines
		File doublePFile = new File("../../data/results/fern_150.dat");
		ArrayList<String[]> doubleLines = getLines(doublePFile);

		// Compute the results and dump them to standard out. They don't need to
		// be directly into a file.
		for (int i = 0; i < singleLines.size(); i++) {
			singleLine = singleLines.get(i);
			doubleLine = doubleLines.get(i);
			// Print the time
			System.out.print(doubleLine[0] + " ");
			// Compute the relative error, (s-d)/d, for each element and print
			// it.
			for (int j = 1; j < singleLines.get(i).length; j++) {
				singleVal = Math.pow(10.0, Double.valueOf(singleLine[j]));
				doubleVal = Math.pow(10.0, Double.valueOf(doubleLine[j]));
				err = Math.abs((singleVal - doubleVal) / doubleVal);
				// If the error is zero, then log(e) ~ -Inf, so cap it at -38.0.
				if (err != 0.0) {
					System.out.print(Math.log10(err) + " ");
				} else {
					System.out.print("-38.0 ");
				}
			}
			System.out.print("\n");
		}

		return;
	}

	/**
	 * This operation parses the CSV or token-separated variable file to get the
	 * populations.
	 * 
	 * @param file
	 *            the files that should be parsed
	 * @return A list of string arrays with an entry in the string array for
	 *         each element and a string array in the list for each time step.
	 */
	private static ArrayList<String[]> getLines(File file) {

		ArrayList<String[]> lines = new ArrayList<String[]>();

		try {
			// Grab the contents of the file
			InputStream stream = new FileInputStream(file);
			BufferedReader reader = new BufferedReader(
					new InputStreamReader(stream));
			String line = null;
			while ((line = reader.readLine()) != null) {
				// Skip lines that are pure comments
				if (!line.startsWith("#")) {
					// Clip the line if it has a comment symbol in it to be
					// everything before the symbol
					if (line.contains("#")) {
						int index = line.indexOf("#");
						line = line.substring(0, index);
					}
					// Clean up any crap on the line
					String[] lineArray = line.trim().split(token);
					// And clean up any crap on each split piece
					for (String element : lineArray) {
						element = element.trim();
					}
					// Put the lines in the list
					lines.add(lineArray);
				}
			}
			// Close the reader
			reader.close();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return lines;
	}

}
