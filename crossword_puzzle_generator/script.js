// script.js

document.addEventListener('DOMContentLoaded', () => {
    const wordLoader = new WordLoader();
    const crosswordUI = new CrosswordUI();
    const crossword = new Crossword(10);

    crosswordUI.initializeEventListeners(crossword, wordLoader);

    // Expose methods to the global scope if needed
    window.checkAnswers = () => crosswordUI.checkAnswers(crossword);
    window.revealAnswers = () => crosswordUI.revealAnswers(crossword);
	
	generateCrosswordOnLoad(crossword, wordLoader, crosswordUI);
});

// Word class
class Word {
    constructor(text, clue) {
        this.text = text.toUpperCase();
        this.clue = clue;
        this.placed = false;
    }
}

// Crossword class
class Crossword {
    constructor(gridSize) {
        this.gridSize = gridSize;
        this.grid = this.initializeGrid(gridSize);
        this.words = [];
        this.clues = { across: [], down: [] };
    }

    initializeGrid(size) {
        return Array.from({ length: size }, () => Array(size).fill(null));
    }

    async generateCrossword(attempts = 20) {
        let bestGrid = null;
        let bestClues = null;
        let maxWordsPlaced = 0;

        for (let attempt = 0; attempt < attempts; attempt++) {
            this.grid = this.initializeGrid(this.gridSize);
            this.clues = { across: [], down: [] };

            shuffleArray(this.words);

            this.words.forEach(word => {
                word.placed = false;
                delete word.row;
                delete word.col;
                delete word.direction;
            });

            this.placeWordsRandomly();

            const wordsPlaced = this.words.filter(word => word.placed).length;

            if (wordsPlaced > maxWordsPlaced) {
                maxWordsPlaced = wordsPlaced;
                bestGrid = JSON.parse(JSON.stringify(this.grid));
                bestClues = JSON.parse(JSON.stringify(this.clues));
            }

            if (wordsPlaced === this.words.length) {
                break;
            }
        }

        this.grid = bestGrid;
        this.clues = bestClues;

    }

    placeWordsRandomly() {
		let clueNumber = 1;
		const directions = ['across', 'down'];

		this.words.forEach((word) => {
			word.placed = false;
		});

		// Place the words in a random order
		shuffleArray(this.words);
		
		for (let word of this.words) {
			let placed = false;
			for (let attempt = 0; attempt < 100 && !placed; attempt++) {
				const direction = directions[Math.floor(Math.random() * directions.length)];
				const row = Math.floor(Math.random() * (this.gridSize - (direction === 'across' ? 0 : word.text.length)));
				const col = Math.floor(Math.random() * (this.gridSize - (direction === 'down' ? 0 : word.text.length)));
				
				if (this.canPlaceWordAt(word, row, col, direction)) {
					placed = this.placeWordAt(word, row, col, direction);
					if (placed) {
						word.placed = true;
						this.clues[direction].push({
							number: clueNumber++,
							clue: word.clue,
							answer: word.text,
							row: row,
							col: col,
							direction: direction,
						});
					}
				}
			}

			if (!placed) {
				console.log(`Failed to place word: ${word.text}`);
			}
		}
	}


    findPositionForWord(word) {
        const placedWords = this.words.filter(w => w.placed);
        placedWords.sort((a, b) => {
            const aCommon = countCommonLetters(word.text, a.text);
            const bCommon = countCommonLetters(word.text, b.text);
            return bCommon - aCommon;
        });

        for (let existingWord of placedWords) {
            for (let i = 0; i < word.text.length; i++) {
                const letter = word.text[i];
                for (let j = 0; j < existingWord.text.length; j++) {
                    if (existingWord.text[j] === letter) {
                        const potentialPositions = this.getPotentialPositions(word, existingWord, i, j);
                        shuffleArray(potentialPositions);
                        for (let pos of potentialPositions) {
                            if (this.canPlaceWordAt(word, pos.row, pos.col, pos.direction)) {
                                return pos;
                            }
                        }
                    }
                }
            }
        }
        return null;
    }

    getPotentialPositions(word, existingWord, letterIndex, existingLetterIndex) {
        const positions = [];
        const existingRow = existingWord.row;
        const existingCol = existingWord.col;
        if (existingWord.direction === 'across') {
            const row = existingRow - letterIndex;
            const col = existingCol + existingLetterIndex;
            positions.push({ row, col, direction: 'down' });
        } else {
            const row = existingRow + existingLetterIndex;
            const col = existingCol - letterIndex;
            positions.push({ row, col, direction: 'across' });
        }
        return positions;
    }

    canPlaceWordAt(word, row, col, direction) {
        if (direction === 'across') {
            if (col < 0 || col + word.text.length > this.gridSize || row < 0 || row >= this.gridSize) {
                return false;
            }
            for (let i = 0; i < word.text.length; i++) {
                const cell = this.grid[row][col + i];
                const letter = word.text[i];
                if (cell) {
                    if (cell.letter !== letter) {
                        return false;
                    }
                }
            }
        } else if (direction === 'down') {
            if (row < 0 || row + word.text.length > this.gridSize || col < 0 || col >= this.gridSize) {
                return false;
            }
            for (let i = 0; i < word.text.length; i++) {
                const cell = this.grid[row + i][col];
                const letter = word.text[i];
                if (cell) {
                    if (cell.letter !== letter) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    placeWordAt(word, row, col, direction) {
		// Check if the starting point of the word is already marked for another word (either across or down)
		if (this.grid[row][col] && this.grid[row][col].isStartOfWord) {
			console.log(`Skipping word '${word.text}' because the cell at (${row}, ${col}) is already a starting point for another word.`);
			return false;  // Indicate that the word couldn't be placed
		}

		if (direction === 'across') {
			for (let i = 0; i < word.text.length; i++) {
				if (!this.grid[row][col + i]) {
					this.grid[row][col + i] = { letter: word.text[i], isStartOfWord: false };
				} else {
					// If a cell already exists, we keep its isStartOfWord flag intact
					this.grid[row][col + i].letter = word.text[i];  // Update the letter
				}
			}
			// Only set the starting point if it's not already marked
			this.grid[row][col].isStartOfWord = true;
		} else {
			for (let i = 0; i < word.text.length; i++) {
				if (!this.grid[row + i][col]) {
					this.grid[row + i][col] = { letter: word.text[i], isStartOfWord: false };
				} else {
					// If a cell already exists, we keep its isStartOfWord flag intact
					this.grid[row + i][col].letter = word.text[i];  // Update the letter
				}
			}
			// Only set the starting point if it's not already marked
			this.grid[row][col].isStartOfWord = true;
		}

		return true;  // Indicate that the word was placed successfully
	}




    findEmptySlots() {
        const slots = [];

        // Find across slots
        for (let row = 0; row < this.gridSize; row++) {
            let startCol = null;
            let length = 0;
            for (let col = 0; col < this.gridSize; col++) {
                const cell = this.grid[row][col];
                if (cell === null) {
                    if (startCol === null) {
                        startCol = col;
                    }
                    length++;
                } else {
                    if (length > 1) {
                        slots.push({
                            row: row,
                            col: startCol,
                            length: length,
                            direction: 'across'
                        });
                    }
                    startCol = null;
                    length = 0;
                }
            }
            if (length > 1) {
                slots.push({
                    row: row,
                    col: startCol,
                    length: length,
                    direction: 'across'
                });
            }
        }

        // Find down slots
        for (let col = 0; col < this.gridSize; col++) {
            let startRow = null;
            let length = 0;
            for (let row = 0; row < this.gridSize; row++) {
                const cell = this.grid[row][col];
                if (cell === null) {
                    if (startRow === null) {
                        startRow = row;
                    }
                    length++;
                } else {
                    if (length > 1) {
                        slots.push({
                            row: startRow,
                            col: col,
                            length: length,
                            direction: 'down'
                        });
                    }
                    startRow = null;
                    length = 0;
                }
            }
            if (length > 1) {
                slots.push({
                    row: startRow,
                    col: col,
                    length: length,
                    direction: 'down'
                });
            }
        }

        console.log('Empty slots found:', slots.length);
        return slots;
    }
}

// WordLoader class
class WordLoader {
    constructor() {
        this.defaultWords = [
			new Word('AGILE', 'A project management methodology focusing on flexibility and collaboration'),
			new Word('SCRUM', 'A framework within Agile for managing complex projects'),
			new Word('JAVASCRIPT', 'A popular programming language primarily used for web development'),
			new Word('PYTHON', 'A high-level programming language known for its simplicity'),
			new Word('JAVA', 'A widely-used programming language, especially for enterprise applications'),
			new Word('OOP', 'A programming paradigm based on the concept of "objects"'),
			new Word('GIT', 'A version control system for tracking changes in source code'),
			new Word('DOCKER', 'A platform for developing, shipping, and running applications in containers'),
			new Word('KUBERNETES', 'An open-source platform for automating the deployment, scaling, and management of containerized applications'),
			new Word('CI', 'Short for Continuous Integration, a DevOps practice'),
			new Word('CD', 'Short for Continuous Delivery or Continuous Deployment, a DevOps practice'),
			new Word('JIRA', 'A tool used for project management and issue tracking'),
			new Word('UML', 'A standardized modeling language for software systems'),
			new Word('TESTING', 'The process of evaluating software to ensure it works as expected'),
			new Word('DEBUGGING', 'The process of identifying and removing errors in software'),
			new Word('ALGORITHM', 'A step-by-step procedure for solving a problem'),
			new Word('DATABASE', 'A structured collection of data stored electronically'),
			new Word('SQL', 'A language used to manage and manipulate databases'),
			new Word('NOSQL', 'A class of databases that are not based on SQL'),
			new Word('APACHE', 'A popular open-source web server software'),
			new Word('NGINX', 'A lightweight and high-performance web server software'),
			new Word('NODEJS', 'A JavaScript runtime built on Chrome\'s V8 JavaScript engine'),
			new Word('REACT', 'A JavaScript library for building user interfaces'),
			new Word('ANGULAR', 'A TypeScript-based open-source web application framework'),
			new Word('VUE', 'A progressive JavaScript framework for building user interfaces'),
			new Word('MVC', 'A software design pattern that separates an application into three components: Model, View, and Controller'),
			new Word('REST', 'An architectural style for designing networked applications'),
			new Word('SOAP', 'A protocol for exchanging structured information in web services'),
			new Word('GRAPHQL', 'A query language for APIs and a runtime for executing queries'),
			new Word('API', 'A set of tools for building software applications'),
			new Word('MICROSERVICES', 'An architectural style that structures an application as a collection of loosely coupled services'),
			new Word('MONOLITH', 'A software application built as a single unit'),
			new Word('CLOUD', 'The on-demand availability of computing resources over the internet'),
			new Word('AWS', 'Amazon\'s cloud computing platform'),
			new Word('AZURE', 'Microsoft\'s cloud computing platform'),
			new Word('GCP', 'Google Cloud Platform'),
			new Word('IAAS', 'Infrastructure as a Service, a cloud computing model'),
			new Word('PAAS', 'Platform as a Service, a cloud computing model'),
			new Word('SAAS', 'Software as a Service, a cloud computing model'),
			new Word('VIRTUALIZATION', 'The creation of a virtual version of a resource, such as a server'),
			new Word('DEVOPS', 'A set of practices that combines software development and IT operations'),
			new Word('SRE', 'Site Reliability Engineering, a discipline that incorporates aspects of software engineering into IT operations'),
			new Word('LOADBALANCER', 'A device that distributes network or application traffic across multiple servers'),
			new Word('CONTAINERIZATION', 'The process of packaging software code with all its dependencies into a container'),
			new Word('CI/CD', 'A practice that involves Continuous Integration and Continuous Deployment'),
			new Word('SERVERLESS', 'A cloud computing model where the cloud provider manages the infrastructure'),
			new Word('BACKEND', 'The server-side of a web application, responsible for business logic and database management'),
			new Word('FRONTEND', 'The client-side of a web application, responsible for the user interface and experience'),
			new Word('HTML', 'The standard markup language for creating web pages'),
			new Word('CSS', 'A style sheet language used for describing the presentation of a document written in HTML'),
			new Word('JSON', 'A lightweight data-interchange format that is easy for humans to read and write'),
			new Word('XML', 'A markup language used to encode documents in a format that is both human-readable and machine-readable'),
			new Word('KERNEL', 'The core of an operating system, responsible for managing system resources'),
			new Word('THREAD', 'The smallest sequence of programmed instructions that can be managed independently'),
			new Word('PROCESS', 'A program in execution'),
			new Word('MUTEX', 'A mechanism used to prevent multiple threads from accessing shared resources simultaneously'),
			new Word('DEADLOCK', 'A situation where two or more processes are unable to proceed because each is waiting for the other to release a resource'),
			new Word('SOFTWAREARCHITECTURE', 'The high-level structure of a software system and the discipline of creating such structures'),
			new Word('SOLID', 'A set of principles for designing software that is easy to maintain and extend'),
			new Word('CAPTHEOREM', 'A principle that states that in a distributed system, you can only have two of the following three: Consistency, Availability, and Partition tolerance'),
			new Word('TDD', 'Test-Driven Development, a software development process where tests are written before the code'),
			new Word('BDD', 'Behavior-Driven Development, a development methodology based on TDD, focusing on the behavior of the system'),
			new Word('VERSIONCONTROL', 'A system for tracking changes to source code over time'),
			new Word('AGGREGATION', 'A relationship between objects in OOP where one object contains others'),
			new Word('COMPOSITION', 'A relationship where one object is composed of one or more other objects in OOP'),
			new Word('INHERITANCE', 'A mechanism in OOP where one class inherits the properties and behaviors of another class'),
			new Word('POLYMORPHISM', 'The ability in OOP to present the same interface for different underlying forms'),
			new Word('ENCAPSULATION', 'The bundling of data and the methods that operate on that data within a single unit, or class, in OOP'),
			new Word('DATASTRUCTURE', 'A way of organizing and storing data so that it can be accessed and modified efficiently'),
			new Word('BINARYTREE', 'A tree data structure in which each node has at most two children'),
			new Word('HASHMAP', 'A data structure that implements an associative array, a structure that can map keys to values'),
			new Word('STACK', 'A data structure that follows Last In, First Out (LIFO) principle'),
			new Word('QUEUE', 'A data structure that follows First In, First Out (FIFO) principle'),
			new Word('LINKEDLIST', 'A linear collection of data elements where each element points to the next'),
			new Word('GRAPH', 'A data structure used to model pairwise relations between objects'),
			new Word('ARRAY', 'A data structure consisting of a collection of elements, each identified by an index'),
			new Word('BIGO', 'A notation used to describe the performance or complexity of an algorithm'),
			new Word('SEARCHALGORITHM', 'An algorithm used to find an item in a data structure'),
			new Word('SORTINGALGORITHM', 'An algorithm that arranges the elements of a list in a certain order'),
			new Word('DFS', 'Depth-First Search, an algorithm for traversing or searching tree or graph data structures'),
			new Word('BFS', 'Breadth-First Search, an algorithm for traversing or searching tree or graph data structures'),
			new Word('HASHING', 'A technique used to uniquely identify a specific object from a group of similar objects'),
			new Word('LOADTESTING', 'A testing process to determine how a system performs under expected load'),
			new Word('STRESSTESTING', 'A testing process to determine the stability of a system by subjecting it to extreme conditions'),
			new Word('UNittesting', 'Testing individual units of source code to determine whether they are fit for use'),
			new Word('INTEGRATIONTESTING', 'Testing multiple components or systems to verify they work together as expected'),
			new Word('SYSTEMTESTING', 'Testing a complete and integrated software system to evaluate its compliance with requirements'),
			new Word('PERFORMANCETESTING', 'Testing to determine how well a software system performs under certain conditions'),
			new Word('AUTOMATION', 'The technique of making an apparatus, process, or system operate automatically'),
			new Word('CODECOVERAGE', 'A measure used to describe the degree to which the source code of a program is executed during testing'),
			new Word('REFACTORING', 'The process of restructuring existing code without changing its external behavior'),
			new Word('DESIGNPATTERN', 'A general reusable solution to a commonly occurring problem within a given context in software design'),
			new Word('LLM', 'Large Language Model, used for understanding and generating text'),
			new Word('GPT', 'Generative Pre-trained Transformer, a type of LLM architecture'),
			new Word('BERT', 'Bidirectional Encoder Representations from Transformers, a transformer-based model'),
			new Word('TRANSFORMER', 'A neural network architecture used in NLP tasks'),
			new Word('NLP', 'Natural Language Processing, a field of AI focused on the interaction between computers and human language'),
			new Word('PROMPT', 'Input text given to an LLM to generate a response'),
			new Word('TOKEN', 'A unit of text used in models, such as words or parts of words'),
			new Word('INFERENCE', 'The process of generating output from an AI model given some input'),
			new Word('FINE-TUNING', 'The process of adapting a pre-trained model to a specific task'),
			new Word('TRANSFERLEARNING', 'A technique where a model trained on one task is reused for a different task'),
			new Word('ZERO-SHOT', 'A model’s ability to perform tasks it wasn’t explicitly trained for'),
			new Word('FEW-SHOT', 'Providing a model with a few examples to guide its responses'),
			new Word('REINFORCEMENTLEARNING', 'A type of learning where agents learn by interacting with their environment'),
			new Word('CHATBOT', 'An AI application designed to simulate human conversation'),
			new Word('GENERATIVEAI', 'AI focused on creating new content, such as text, images, or music'),
			new Word('DIALOGSYSTEM', 'A system designed to converse with humans using natural language'),
			new Word('BLOOM', 'An open-source large-scale language model trained on multiple languages'),
			new Word('DALLE', 'A generative model that creates images from text descriptions'),
			new Word('CLIP', 'A model that understands images and text, used in multimodal tasks'),
			new Word('PROMPTENGINEERING', 'The craft of designing input prompts to guide AI models towards desired outputs'),
			new Word('UNSUPERVISEDLEARNING', 'A type of machine learning that infers patterns from data without labeled responses'),
			new Word('SUPERVISEDLEARNING', 'A type of machine learning where the model learns from labeled training data'),
			new Word('LSTM', 'Long Short-Term Memory, a type of recurrent neural network'),
			new Word('SEQ2SEQ', 'Sequence to Sequence, a model that maps input sequences to output sequences'),
			new Word('ATTENTION', 'A mechanism that allows models to focus on important parts of the input'),
			new Word('MULTIMODALAI', 'AI systems that can process and combine multiple types of data like text and images'),
			new Word('DATAAUGMENTATION', 'A technique used to increase the diversity of training data by altering existing data'),
			new Word('PRETRAINING', 'The process of training a model on a large dataset before fine-tuning it for a specific task'),
			new Word('HALLUCINATION', 'When an AI model generates plausible but incorrect or nonsensical information'),
			new Word('EMBEDDINGS', 'Low-dimensional representations of words or tokens in a model'),
			new Word('WORD2VEC', 'A technique used to create word embeddings from large datasets'),
			new Word('CONTEXTWINDOW', 'The range of tokens or words that a model can consider when generating a response'),
			new Word('DETR', 'A model designed for object detection in images using transformers'),
			new Word('SELFATTENTION', 'A mechanism that allows each part of an input to weigh its relevance to other parts'),
			new Word('BIDIRECTIONAL', 'Processing input data in both forward and backward directions in neural networks'),
			new Word('POSENCODING', 'Positional Encoding, a technique used in transformers to inject sequence order information'),
			new Word('LATENTSPACE', 'A high-dimensional space where data representations are stored during model training'),
			new Word('VAE', 'Variational Autoencoder, a generative model that learns to encode and decode data'),
			new Word('GAN', 'Generative Adversarial Network, a model that generates new data via adversarial training'),
			new Word('DISCRIMINATOR', 'Part of a GAN responsible for distinguishing between real and generated data'),
			new Word('GENERATOR', 'Part of a GAN that creates new data to fool the discriminator'),
			new Word('STYLEGAN', 'A type of GAN that generates high-quality images with controllable style features'),
			new Word('VQGAN', 'A generative model that combines GANs with vector quantization for image synthesis'),
			new Word('HUGGINGFACE', 'A platform and library for working with transformers and LLMs'),
			new Word('SPARSEATTENTION', 'A variation of attention mechanisms designed to reduce computation cost'),
			new Word('OPTIMIZER', 'An algorithm that adjusts the parameters of a model to minimize loss during training'),
			new Word('BATCHSIZE', 'The number of samples processed before the model is updated'),
			new Word('ADAM', 'A popular optimization algorithm used in training neural networks'),
			new Word('BLOOMFILTER', 'A probabilistic data structure used to test whether an element is part of a set'),
			new Word('BEAMSEARCH', 'A decoding algorithm used to generate text in language models'),
			new Word('TEMPERATURE', 'A hyperparameter that controls the randomness of predictions in LLMs'),
			new Word('LOSSFUNCTION', 'A function that measures how well a model’s predictions match the expected output'),
			new Word('PERPLEXITY', 'A measurement used to evaluate language models, lower perplexity indicates better predictions'),
			new Word('F1SCORE', 'A metric used to evaluate a model’s accuracy based on precision and recall'),
			new Word('BLEU', 'A metric for evaluating the quality of text generated by machine translation models'),
			new Word('ROUGE', 'A set of metrics used to evaluate automatic summarization and translation models'),
			new Word('EXPLAINABLEAI', 'AI systems that provide insights into how decisions are made'),
			new Word('SOFTMAX', 'A function used to turn raw output into probabilities in classification models'),
			new Word('LANGCHAIN', 'A framework for developing complex applications with LLMs'),
			new Word('ZERO_SHOTLEARNING', 'A model’s ability to generalize to unseen classes during inference'),
			new Word('LORA', 'Low-Rank Adaptation, used to fine-tune large models with fewer parameters'),
			new Word('PROMPTTUNING', 'Modifying prompts to improve model performance without retraining'),
			new Word('ALPACA', 'A fine-tuned LLaMA model built for instruction-following tasks'),
			new Word('AUTOENCODER', 'A neural network designed to encode and then decode data, often used for compression'),
			new Word('CLUSTERING', 'A type of unsupervised learning where data is grouped based on similarity'),
			new Word('FID', 'Fréchet Inception Distance, a metric used to evaluate the quality of generated images'),
			new Word('SAMPLING', 'The process of generating new data points from a trained model’s output distribution'),
			new Word('DIALOGPT', 'A large-scale generative model for open-domain conversation'),
			new Word('PLM', 'Pretrained Language Model, the basis of many LLM architectures'),
			new Word('EPOCH', 'One complete pass through the training data during model training'),
			new Word('METALEARNING', 'A form of learning that allows models to adapt to new tasks with minimal data'),
			new Word('RELU', 'Rectified Linear Unit, a popular activation function in neural networks'),
			new Word('SOFTPROMPT', 'A method that uses soft embeddings to improve prompt-based learning'),
			new Word('SYNTHETICDATA', 'Artificially generated data used to augment real datasets during model training'),
			new Word('WORDVECTORS', 'A method of representing words as vectors in continuous space'),
			new Word('GRADIENTDESCENT', 'An optimization technique used to minimize the loss function'),
			new Word('MASKEDLANGUAGE', 'A task in which parts of a sentence are hidden and the model must predict the missing words'),
			new Word('HIDDENLAYER', 'An intermediate layer in a neural network between input and output'),
			new Word('FINETUNING', 'The process of updating a pre-trained model on a new, smaller dataset'),
			new Word('POSITIVEPAIR', 'Two examples that are semantically similar, used in contrastive learning'),
			new Word('NEGATIVEPAIR', 'Two examples that are semantically different, used in contrastive learning'),
			new Word('MARGINLOSS', 'A loss function used in ranking problems to push dissimilar examples apart'),
			new Word('MULTITASKLEARNING', 'A method where a model learns multiple tasks at once to improve performance'),
			new Word('CONTINUALLEARNING', 'A method that allows a model to learn continuously without forgetting previous tasks'),
			new Word('FEWSHOTLEARNING', 'A form of learning where models are trained on very few examples'),
			new Word('TOKENIZATION', 'The process of splitting text into individual tokens for a model'),
			new Word('AUTOREGRESSIVE', 'A model that generates text one token at a time by conditioning on previously generated tokens'),
			new Word('CURIE', 'An OpenAI GPT-3 model optimized for speed and efficiency'),
			new Word('DAVINCI', 'OpenAI’s most powerful GPT-3 model, used for complex tasks'),
			new Word('CODER', 'A specialized LLM fine-tuned for code generation tasks'),
			new Word('PLUGINDESIGN', 'A system that enables LLMs to extend their capabilities with external tools'),
			new Word('CHAINOFTHOUGHT', 'A technique where the model generates intermediate reasoning steps before answering a question'),
			new Word('RAG', 'Retrieval-Augmented Generation, a method combining retrieval and generative models for factual generation'),
			new Word('T5', 'Text-To-Text Transfer Transformer, a model that frames NLP tasks as text-to-text problems'),
			new Word('FLAN', 'A fine-tuned T5 model built for instruction-following'),
			new Word('HYPERPARAMETER', 'A parameter whose value is set before training the model and affects training efficiency'),
			new Word('SPARSEMODEL', 'A model that uses sparsity to reduce the number of active neurons, improving efficiency'),
			new Word('GRADIENTCLIPPING', 'A technique used to prevent exploding gradients during training'),
			new Word('XLM', 'A multilingual transformer model built for cross-lingual tasks'),
			new Word('CONTRASTIVELEARNING', 'A technique used to learn representations by comparing positive and negative pairs'),
			new Word('LATENTDIFFUSION', 'A generative technique for creating high-quality images by sampling from a latent space'),
			new Word('DEEPDREAM', 'A neural network algorithm that enhances patterns in images to create dream-like visuals'),
			new Word('UNET', 'A model architecture often used in image segmentation tasks'),
			new Word('OPENAI', 'A research organization that develops and promotes friendly AI'),
			new Word('COHERENCE', 'The logical consistency and relevance of a generated text'),
			new Word('REPHRASING', 'A task where the model generates alternative ways to express the same idea'),
			new Word('SUMMARIZATION', 'A task where the model generates a concise summary of a given text'),
			new Word('QUESTIONANSWERING', 'A task where the model answers questions based on a given context'),
			new Word('ETHICSOFAI', 'The study of moral implications and fairness in artificial intelligence systems'),
			new Word('MODELDRIFT', 'When a model’s performance degrades due to changing data patterns over time'),
			new Word('EXPERIENCEREPLAY', 'A technique used in reinforcement learning to store and reuse past interactions'),
			new Word('SEQ2SEQ', 'A model used for translating one sequence of text into another'),
			new Word('GRAPHTRANSFORMER', 'A transformer model adapted for tasks involving graph data'),
			new Word('MULTITASK', 'A method where the model performs multiple tasks simultaneously'),
			new Word('LONGFORM', 'A task where the model generates longer passages of coherent text'),
			new Word('BIMODAL', 'Models that handle two different types of data, like text and images'),
			new Word('HYBRIDMODEL', 'A model combining different architectures or techniques to achieve better results'),
			new Word('NEURALNETWORK', 'A network of artificial neurons used for tasks like classification and regression'),
			new Word('KNOWLEDGEGRAPH', 'A graph-based representation of real-world entities and their relationships'),
			new Word('SCALABILITY', 'The capability of a model to handle growing amounts of data or tasks effectively'),
			new Word('TRANSFERLEARNING', 'The practice of reusing a pre-trained model on a new task or domain'),
			new Word('SIMILARITYSEARCH', 'A task where the model finds items most similar to a given query'),
			new Word('HYPERGRAPH', 'A graph structure where edges can connect multiple nodes simultaneously'),
			new Word('ZERO_SHOTGENERATION', 'The ability to generate content on tasks it was not explicitly trained for'),
			new Word('DIALOGICAI', 'AI systems focused on human-like conversations and dialogue management'),
			new Word('TEXTAUGMENTATION', 'Using techniques to enhance or modify text data for training'),
			new Word('SENTIMENTANALYSIS', 'A task where the model classifies text based on emotional tone or opinion'),
			new Word('KNOWLEDGEDISTILLATION', 'A method of transferring knowledge from a larger model to a smaller one'),
			new Word('BILSTM', 'Bidirectional LSTM, a neural network that processes data in both forward and backward directions')
		];

    }

    loadWordsFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = event => {
                const content = event.target.result;
                const words = this.parseWordsFromContent(content);
                resolve(words);
            };
            reader.onerror = () => {
                reject(reader.error);
            };
            reader.readAsText(file);
        });
    }

    parseWordsFromContent(content) {
        const words = [];
        const lines = content.split(/\r?\n/);
        lines.forEach(line => {
            const [clue, answer] = line.split(',');
            if (clue && answer) {
                words.push(new Word(answer.trim(), clue.trim()));
            }
        });
        return words;
    }

    getDefaultWords() {
        return [...this.defaultWords];
    }
}

// CrosswordUI class
class CrosswordUI {
    constructor() {
        this.gridSizeInput = document.getElementById('grid-size');
        this.wordFileInput = document.getElementById('word-file');
        this.generateButton = document.getElementById('generate-button');
        this.crosswordContainer = document.getElementById('crossword-container');
        this.crosswordElement = document.getElementById('crossword');
        this.allCluesElement = document.getElementById('all-clues');
    }

    initializeEventListeners(crossword, wordLoader) {
        this.generateButton.addEventListener('click', () => {
            const gridSize = parseInt(this.gridSizeInput.value);
            if (isNaN(gridSize) || gridSize < 5 || gridSize > 15) {
                alert('Please enter a valid grid size between 5 and 15.');
                return;
            }

            crossword.gridSize = gridSize;
            
            crossword.words = wordLoader.getDefaultWords();
            this.generateCrossword(crossword);
        });
    }

    async generateCrossword(crossword) {
        await crossword.generateCrossword();
        this.crosswordContainer.style.display = 'block';
        this.createGrid(crossword);
        this.displayClues(crossword);
    }

    createGrid(crossword) {
        const { gridSize, grid, clues } = crossword;
        this.crosswordElement.innerHTML = '';
        this.crosswordElement.style.gridTemplateColumns = `repeat(${gridSize}, 40px)`;
        const totalGridWidth = gridSize * 40 + (gridSize - 1) * 1;
        this.crosswordElement.style.width = `${totalGridWidth}px`;

        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
                const cellDiv = document.createElement('div');
                const cellData = grid[row][col];

                if (!cellData) {
                    cellDiv.classList.add('cell', 'black-cell');
                } else {
                    cellDiv.classList.add('cell');
                    const input = document.createElement('input');
                    input.setAttribute('maxlength', '1');
                    input.setAttribute('data-row', row);
                    input.setAttribute('data-col', col);
					input.style.caretColor = 'black';
					
                    input.oninput = () => {
                        input.value = input.value.toUpperCase();
						input.style.color = 'black';
						input.style.caretColor = 'black';
						
						this.moveToNextCell(row, col, crossword);
                    };
                    input.addEventListener('focus', event => this.handleCellFocus(event, crossword));
                    cellDiv.appendChild(input);

                    if (cellData.isStartOfWord) {
                        const numberDiv = document.createElement('div');
                        numberDiv.classList.add('cell-number');
                        const clueNumber = this.getClueNumberAt(row, col, clues);
                        numberDiv.textContent = clueNumber;
                        cellDiv.appendChild(numberDiv);
                    }
                }
                this.crosswordElement.appendChild(cellDiv);
            }
        }

        this.crosswordElement.addEventListener('click', event => this.handleCrosswordClick(event));
    }

	moveToNextCell(currentRow, currentCol, crossword) {
		// Get the current word that this cell belongs to
		const word = this.findWordAtPosition(currentRow, currentCol, crossword.clues);
		
		if (word) {
			for (let i = 0; i < word.answer.length; i++) {
				let cellRow = word.row;
				let cellCol = word.col;

				if (word.direction === 'across') {
					cellCol += i;
				} else {
					cellRow += i;
				}

				// If this is the current cell, find the next one
				if (cellRow === currentRow && cellCol === currentCol) {
					if (i < word.answer.length - 1) {
						// Move to the next cell in the word
						const nextRow = word.direction === 'across' ? currentRow : currentRow + 1;
						const nextCol = word.direction === 'across' ? currentCol + 1 : currentCol;
						const nextInput = document.querySelector(`input[data-row="${nextRow}"][data-col="${nextCol}"]`);
						if (nextInput) {
							nextInput.focus(); // Set focus to the next cell
						}
					}
					break; // Exit the loop after finding the current cell
				}
			}
		}
	}


	displayClues(crossword) {
		const { clues } = crossword;
		this.allCluesElement.innerHTML = ''; // Clear existing clues

		// Combine across and down clues into a single list
		const combinedClues = [...clues.across, ...clues.down];

		// Sort the combined clues by their clue number
		combinedClues.sort((a, b) => a.number - b.number);

		// Generate the list and add to the HTML
		combinedClues.forEach(clue => {
			const li = document.createElement('li');
			const clueType = clue.direction === 'across' ? 'Across' : 'Down';

			// Add the clue number in bold and the clue text
			li.innerHTML = `<strong>${clue.number}.</strong> ${clue.clue}`;

			this.allCluesElement.appendChild(li);
		});
	}


    getClueNumberAt(row, col, clues) {
        for (let clue of [...clues.across, ...clues.down]) {
            if (clue.row === row && clue.col === col) {
                return clue.number;
            }
        }
        return '';
    }

    checkAnswers(crossword) {
        const inputs = document.querySelectorAll('.crossword input');
        
        inputs.forEach(input => {
            const row = parseInt(input.getAttribute('data-row'));
            const col = parseInt(input.getAttribute('data-col'));
            const cell = crossword.grid[row][col];
            
            if (cell && cell.letter === input.value.toUpperCase()) {
                input.classList.remove('incorrect');
                input.style.color = 'black';  // Reset text color to black for correct letters
            } else {
                input.classList.add('incorrect');
                input.style.color = 'red';    // Text color red for incorrect letters
            }
            input.style.caretColor = 'black';  // Ensure caret (cursor) color is always black
        });
	}

    revealAnswers(crossword) {
        const inputs = document.querySelectorAll('.crossword input');
        inputs.forEach(input => {
            const row = parseInt(input.getAttribute('data-row'));
            const col = parseInt(input.getAttribute('data-col'));
            const cell = crossword.grid[row][col];
            if (cell) {
                input.value = cell.letter;
                input.classList.add('correct');
				input.style.color = 'black';  // Ensure the text color is set to black
                input.style.caretColor = 'black';
            }
        });
    }

    handleCellFocus(event, crossword) {
		const row = parseInt(event.target.getAttribute('data-row'));
		const col = parseInt(event.target.getAttribute('data-col'));

		// Check if the clicked cell is the start of a word
		const cell = crossword.grid[row][col];
		if (cell && cell.isStartOfWord) {
			// Highlight the word that starts at this cell (across or down)
			this.highlightWordAtStart(row, col, crossword);
		} else {
			// If it's not the start of a word, highlight the word this cell belongs to
			this.highlightWordAtPosition(row, col, crossword);
		}
	}


	highlightWordAtPosition(row, col, crossword) {
		this.removeAllHighlights();
		const word = this.findWordAtPosition(row, col, crossword.clues);
		if (word) {
			this.highlightWord(word);
		}
	}

	
	highlightWordAtStart(row, col, crossword) {
		this.removeAllHighlights();

		// Find the word that starts at this position (across or down)
		const wordAcross = crossword.clues.across.find(clue => clue.row === row && clue.col === col);
		const wordDown = crossword.clues.down.find(clue => clue.row === row && clue.col === col);

		// Check for across word
		if (wordAcross) {
			console.log(`Clue Number (Across): ${wordAcross.number}, Word: ${wordAcross.answer}`);
			this.highlightWord(wordAcross); // Highlight the across word
		} 

		// Check for down word
		if (wordDown) {
			console.log(`Clue Number (Down): ${wordDown.number}, Word: ${wordDown.answer}`);
			this.highlightWord(wordDown); // Highlight the down word
		}
	}



	highlightWord(word) {
		for (let i = 0; i < word.answer.length; i++) {
			let cellRow = word.row;
			let cellCol = word.col;
			if (word.direction === 'across') {
				cellCol += i;
			} else {
				cellRow += i;
			}
			const input = document.querySelector(`input[data-row="${cellRow}"][data-col="${cellCol}"]`);
			if (input) {
				input.parentElement.classList.add('highlight');
			}
		}
	}

    handleCrosswordClick(event) {
        if (event.target.tagName !== 'INPUT') {
            this.removeAllHighlights();
        }
    }
	
	findWordAtStart(row, col, clues) {
        return clues.across.find(clue => clue.row === row && clue.col === col) ||
               clues.down.find(clue => clue.row === row && clue.col === col);
    }

    removeAllHighlights() {
        const highlightedCells = document.querySelectorAll('.cell.highlight');
        highlightedCells.forEach(cell => cell.classList.remove('highlight'));
    }

    findWordAtPosition(row, col, clues) {
        return [...clues.across, ...clues.down].find(clue => {
            if (clue.direction === 'across') {
                return row === clue.row && col >= clue.col && col < clue.col + clue.answer.length;
            } else {
                return col === clue.col && row >= clue.row && row < clue.row + clue.answer.length;
            }
        });
    }
}

// Utility functions
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

function countCommonLetters(word1, word2) {
    let count = 0;
    for (let char of word1) {
        if (word2.includes(char)) count++;
    }
    return count;
}


function printCrossword() {
    // Get the crossword and clues elements
    const crossword = document.getElementById('crossword');
    const allClues = document.getElementById('all-clues');

    // Create a new window for printing
    const printWindow = window.open('', '', 'width=800,height=600');
    printWindow.document.write('<html><head><title>Print Crossword</title>');

    // Add CSS to style the print page, ensuring gridlines and black cells are properly visible
    printWindow.document.write(`
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }
            .crossword {
                display: grid;
                grid-gap: 1px;
                grid-template-columns: ${getGridTemplateColumns(crossword)};
                margin-bottom: 20px;
            }
            .cell {
                width: 40px;
                height: 40px;
                text-align: center;
                vertical-align: middle;
                border: 1px solid #333; /* Ensure visible gridlines */
                font-size: 18px;
            }
            .black-cell {
                background-color: #333 !important; /* Ensure black cells are filled */
                border: 1px solid #333; /* Match border with background to blend */
            }
            .clues {
                margin-top: 20px;
            }
            .clues h3 {
                margin-bottom: 5px;
            }
            .clues ul {
                list-style: none;
                padding-left: 0;
            }
            .clues li {
                margin-bottom: 5px;
            }
            strong {
                font-weight: bold;
            }
        </style>
    `);

    // Add the crossword grid to the print page
    printWindow.document.write('</head><body>');
    printWindow.document.write('<div class="crossword">');

    // Append each cell in the crossword grid to the print page
    crossword.querySelectorAll('.cell').forEach(cell => {
        const newCell = cell.cloneNode(true);
        const input = newCell.querySelector('input');
        if (input) {
            newCell.removeChild(input); // Remove input field for printing
            const letter = input.value.toUpperCase();
            if (letter) {
                newCell.textContent = letter; // Place the letter in the cell
            }
        }

        // Ensure the black-cell styling is applied correctly using inline styles
        if (cell.classList.contains('black-cell')) {
            newCell.style.backgroundColor = '#333';  // Apply background color directly
            newCell.style.border = '1px solid #333'; // Apply border directly
        }
        
        printWindow.document.body.querySelector('.crossword').appendChild(newCell);
    });

    printWindow.document.write('</div>');

    // Add the clues to the print page
    printWindow.document.write('<div class="clues"><h3>Clues</h3><ul>');
    allClues.querySelectorAll('li').forEach(clue => {
        printWindow.document.write('<li>' + clue.innerHTML + '</li>');
    });
    printWindow.document.write('</ul></div>');

    // Close the HTML and trigger print
    printWindow.document.write('</body></html>');
    printWindow.document.close();
    printWindow.focus();
    printWindow.print();
    printWindow.close();
}

// Utility function to get grid template columns for printing
function getGridTemplateColumns(crossword) {
    const gridColumns = window.getComputedStyle(crossword).getPropertyValue('grid-template-columns');
    return gridColumns;
}

function generateCrosswordOnLoad(crossword, wordLoader, crosswordUI) {
    const gridSize = parseInt(crosswordUI.gridSizeInput.value) || 10;

    // Set the grid size for the crossword
    crossword.gridSize = gridSize;

    // Load the default words (or you can modify this to load from a file if needed)
    crossword.words = wordLoader.getDefaultWords();

    // Generate the crossword and update the UI
    crosswordUI.generateCrossword(crossword);
}
