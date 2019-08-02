class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

let tanh = new ActivationFunction(
  x => Math.tanh(x),
  y => 1 - (y * y)
);

let relu = new ActivationFunction(
  x => Math.max(0,x),
  y => (y>0)?1:0
)


class NeuralNetwork {
  /*
  * if first argument is a NeuralNetwork the constructor clones it
  * USAGE: cloned_nn = new NeuralNetwork(to_clone_nn);
  */
  constructor(in_nodes, hid_nodes, out_nodes) {
    if (in_nodes instanceof NeuralNetwork) {
      let a = in_nodes;
      this.input_dim = a.input_dim;
      this.hidden_dim = a.hidden_dim;
      this.output_dim = a.output_dim;

      this.weights_ih = a.weights_ih.copy();
      this.weights_ho = a.weights_ho.copy();

      this.bias_h = a.bias_h.copy();
      this.bias_o = a.bias_o.copy();
      this.learning_rate = a.learning_rate
    } else {
      this.input_dim = in_nodes;
      this.hidden_dim = hid_nodes;
      this.output_dim = out_nodes;

      this.weights_ih = new Matrix(this.hidden_dim, this.input_dim);
      this.weights_ho = new Matrix(this.output_dim, this.hidden_dim);
      this.weights_ih.randomize();
      this.weights_ho.randomize();

      this.bias_h = new Matrix(this.hidden_dim, 1);
      this.bias_o = new Matrix(this.output_dim, 1);
      this.bias_h.randomize();
      this.bias_o.randomize();
    }

    // TODO: copy these as well
    this.learning_rate = 0.1
    this.activation_function = sigmoid
  }

  predict(input_array) {
    this.inputs = Matrix.fromArray(input_array);
    this.hidden = Matrix.multiply(this.weights_ih, this.inputs);    // Generating the Hidden Outputs
    this.hidden.add(this.bias_h);
    this.hidden.map(this.activation_function.func);    // activation function!

    this.outputs = Matrix.multiply(this.weights_ho, this.hidden);    // Generating the output's output!
    this.outputs.add(this.bias_o);
    this.outputs.map(this.activation_function.func);

    return this.outputs.toArray();
  }

  setActivationFunction(func = sigmoid) {
    this.activation_function = func;
  }

  train(input_array, target_array) {
    this.predict(input_array)

    let targets = Matrix.fromArray(target_array);    // Convert array to matrix object

    let output_errors = Matrix.subtract(targets, this.outputs);    // Calculate the ERROR = TARGETS - OUTPUTS

    let gradients = Matrix.map(this.outputs, this.activation_function.dfunc);    // Calculate gradient = outputs * (1 - outputs);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    let hidden_T = Matrix.transpose(this.hidden);    // Calculate deltas
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

    this.weights_ho.add(weight_ho_deltas);    // Adjust the weights by deltas
    this.bias_o.add(gradients);    // Adjust the bias by its deltas (which is just the gradients)

    let weight_ho_T = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(weight_ho_T, output_errors);    // Calculate the hidden layer errors

    let hidden_gradient = Matrix.map(this.hidden, this.activation_function.dfunc);    // Calculate hidden gradient
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    let inputs_T = Matrix.transpose(this.inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);    // Calcuate input->hidden deltas

    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hidden_gradient);    // Adjust the bias by its deltas (which is just the gradients)
  }

  serialize() {
    return JSON.stringify(this);
  }

  static instantiate(data) {
    /*
      If data is another NeuralNetwork then this will return a copy.
      If data is a string then we assume it the JSONified version of a NeuralNetwork and parse it as such.
      i.e. to copy of NeuralNetwork do this:
        new_nn = NeuralNetwork.instantiate(nn)
      i.e. to restore a saved NeuralNetwork do this:
        nn = NeuralNetwork.instantiate(json)
    */
    if (typeof data == 'string') {
      data = JSON.parse(data);
    }

    let nn = new NeuralNetwork(data.input_dim, data.hidden_dim, data.output_dim);

    nn.weights_ih = Matrix.deserialize(data.weights_ih);
    nn.weights_ho = Matrix.deserialize(data.weights_ho);
    nn.bias_h = Matrix.deserialize(data.bias_h);
    nn.bias_o = Matrix.deserialize(data.bias_o);
    nn.learning_rate = data.learning_rate;

    return nn;
  }

  // Accept an arbitrary function for mutation
  mutate(func) {
    this.weights_ih.map(func);
    this.weights_ho.map(func);
    this.bias_h.map(func);
    this.bias_o.map(func);
  }
}
