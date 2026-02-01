import OpenAI from "openai";
import { Readable } from "stream";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export const config = {
  api: {
    bodyParser: false, // Disable default body parser for file uploads
  },
};

// IELTS Speaking Band Score System Prompt
const IELTS_EXAMINER_PROMPT = `You are an expert IELTS speaking examiner with 10+ years of experience. Analyze this IELTS speaking response and provide detailed band scores and feedback.

IMPORTANT: You MUST respond with ONLY valid JSON in this exact format (no markdown, no code blocks, just pure JSON):

{
  "overallBand": 7.0,
  "scores": {
    "fluencyCoherence": 7.0,
    "lexicalResource": 7.0,
    "grammaticalRange": 7.0,
    "pronunciation": 7.0
  },
  "feedback": {
    "summary": "One sentence overall assessment of the response",
    "fluencyAnalysis": {
      "score": 7.0,
      "strengths": ["Specific strength with quoted example"],
      "improvements": ["Specific area to improve with quoted example"],
      "details": "Analysis of speech rate, pauses, hesitations, discourse markers, logical flow"
    },
    "lexicalAnalysis": {
      "score": 7.0,
      "strengths": ["Specific strength with quoted example"],
      "improvements": ["Specific area to improve with quoted example"],
      "details": "Analysis of vocabulary range, topic-specific words, collocations, word choice"
    },
    "grammarAnalysis": {
      "score": 7.0,
      "strengths": ["Specific strength with quoted example"],
      "improvements": ["Specific area to improve with quoted example"],
      "details": "Analysis of sentence structures, tenses, complex grammar, accuracy"
    },
    "pronunciationAnalysis": {
      "score": 7.0,
      "strengths": ["Specific strength"],
      "improvements": ["Specific area to improve"],
      "details": "Analysis of clarity, word stress, intonation patterns"
    },
    "quotedExamples": {
      "effective": ["When you said '[exact quote]', this demonstrated [skill] because..."],
      "needsWork": ["The phrase '[exact quote]' could be improved by..."]
    },
    "nextBandTips": [
      "Specific actionable tip 1 to reach the next band level",
      "Specific actionable tip 2 to reach the next band level",
      "Specific actionable tip 3 to reach the next band level"
    ],
    "targetBand": 7.5
  }
}

BAND SCORE GUIDELINES (use official IELTS descriptors):

FLUENCY AND COHERENCE:
- Band 9: Speaks fluently with only rare hesitation, fully coherent with sophisticated discourse markers
- Band 7: Speaks at length without noticeable effort, uses discourse markers flexibly
- Band 5: Can keep going but with frequent repetition, self-correction, slow speech

LEXICAL RESOURCE:
- Band 9: Wide vocabulary with precise meanings, natural collocations, idiomatic expressions
- Band 7: Flexible vocabulary, some less common words, occasional errors in word choice
- Band 5: Limited vocabulary, manages familiar topics, noticeable errors in word formation

GRAMMATICAL RANGE AND ACCURACY:
- Band 9: Wide range of structures, full flexibility, rare minor errors
- Band 7: Range of complex structures, frequent error-free sentences, good control
- Band 5: Basic sentence forms, limited complex structures, frequent errors

PRONUNCIATION:
- Band 9: Full range of features, effortless to understand, L1 accent has no effect
- Band 7: Easy to understand, shows all positive features, some mispronunciation
- Band 5: Generally intelligible, limited range of features, mispronunciation causes difficulty

CRITICAL INSTRUCTIONS:
1. Be honest and accurate - don't inflate scores
2. Quote specific phrases from the transcript in your feedback
3. Base scores strictly on IELTS band descriptors
4. Calculate overall band as average of 4 scores (rounded to nearest 0.5)
5. Provide concrete, actionable advice for improvement
6. Focus on what IELTS examiners actually assess`;

// Helper to parse multipart form data
async function parseMultipartForm(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', () => {
      try {
        const buffer = Buffer.concat(chunks);
        const boundary = req.headers['content-type']?.split('boundary=')[1];

        if (!boundary) {
          return reject(new Error('No boundary found'));
        }

        const parts = buffer.toString('binary').split(`--${boundary}`);
        const result = { fields: {}, files: {} };

        for (const part of parts) {
          if (part.includes('Content-Disposition')) {
            const nameMatch = part.match(/name="([^"]+)"/);
            const filenameMatch = part.match(/filename="([^"]+)"/);

            if (nameMatch) {
              const name = nameMatch[1];
              const contentStart = part.indexOf('\r\n\r\n') + 4;
              const contentEnd = part.lastIndexOf('\r\n');
              const content = part.substring(contentStart, contentEnd);

              if (filenameMatch) {
                // It's a file
                const filename = filenameMatch[1];
                result.files[name] = {
                  filename,
                  data: Buffer.from(content, 'binary')
                };
              } else {
                // It's a field
                result.fields[name] = content;
              }
            }
          }
        }

        resolve(result);
      } catch (error) {
        reject(error);
      }
    });
    req.on('error', reject);
  });
}

// Analyze speech metrics from Whisper word timestamps
function analyzeMetrics(words, duration) {
  if (!words || words.length === 0) {
    return {
      wordsPerMinute: 0,
      averagePauseDuration: 0,
      longestPause: 0,
      fillerWordCount: 0,
      pacingVariation: 'unknown'
    };
  }

  const fillerWords = ['um', 'uh', 'like', 'you know', 'so', 'basically', 'actually', 'literally'];
  let fillerCount = 0;
  const pauses = [];

  // Count filler words
  words.forEach(word => {
    const wordText = word.word.toLowerCase().trim();
    if (fillerWords.includes(wordText)) {
      fillerCount++;
    }
  });

  // Calculate pauses between words
  for (let i = 1; i < words.length; i++) {
    const pause = words[i].start - words[i - 1].end;
    if (pause > 0.2) { // Pauses longer than 200ms
      pauses.push(pause);
    }
  }

  // Calculate words per minute
  const wordsPerMinute = Math.round((words.length / duration) * 60);

  // Analyze pacing variation
  let pacingVariation = 'steady';
  if (pauses.length > words.length * 0.3) {
    pacingVariation = 'halting';
  } else if (pauses.length < words.length * 0.1 && wordsPerMinute > 150) {
    pacingVariation = 'rushed';
  }

  return {
    wordsPerMinute,
    averagePauseDuration: pauses.length > 0 ? (pauses.reduce((a, b) => a + b, 0) / pauses.length).toFixed(2) : 0,
    longestPause: pauses.length > 0 ? Math.max(...pauses).toFixed(2) : 0,
    fillerWordCount: fillerCount,
    pauseCount: pauses.length,
    pacingVariation
  };
}

export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle OPTIONS preflight request
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const contentType = req.headers['content-type'] || '';

    // Handle audio file upload
    if (contentType.includes('multipart/form-data')) {
      const { files, fields } = await parseMultipartForm(req);
      const audioFile = files.audio;

      if (!audioFile) {
        return res.status(400).json({ error: 'No audio file provided' });
      }

      console.log(`Processing audio file: ${audioFile.filename}, size: ${audioFile.data.length} bytes`);

      // Step 1: Transcribe with Whisper (with word-level timestamps)
      const transcription = await client.audio.transcriptions.create({
        file: new File([audioFile.data], audioFile.filename, { type: 'audio/webm' }),
        model: 'whisper-1',
        response_format: 'verbose_json',
        timestamp_granularities: ['word']
      });

      const transcript = transcription.text;
      const words = transcription.words || [];
      const duration = fields.duration ? parseFloat(fields.duration) : transcription.duration || 0;

      console.log(`Transcription complete. Words: ${words.length}, Duration: ${duration}s`);

      // Step 2: Analyze speech metrics
      const metrics = analyzeMetrics(words, duration);

      console.log('Metrics:', metrics);

      // Step 3: Generate IELTS band scores and feedback with GPT
      const metricsText = `
SPEECH METRICS (use these for fluency analysis):
- Speaking pace: ${metrics.wordsPerMinute} words per minute (${
        metrics.wordsPerMinute < 120 ? 'slow - may indicate hesitation' :
        metrics.wordsPerMinute > 160 ? 'fast - may affect clarity' :
        'moderate - good pace'
      })
- Pacing variation: ${metrics.pacingVariation}
- Notable pauses: ${metrics.pauseCount} pauses detected
- Average pause duration: ${metrics.averagePauseDuration}s
- Longest pause: ${metrics.longestPause}s
- Filler words: ${metrics.fillerWordCount} instances ("um", "uh", "like", etc.)
- Total words: ${words.length}
- Speech duration: ${duration}s
`;

      const question = fields.question || 'IELTS Speaking Question';
      const part = fields.part || '1';

      const completion = await client.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [
          {
            role: 'system',
            content: IELTS_EXAMINER_PROMPT
          },
          {
            role: 'user',
            content: `IELTS Speaking Part ${part} Question: "${question}"

${metricsText}

TRANSCRIPT OF RESPONSE:
"${transcript}"

Analyze this IELTS speaking response and provide band scores with detailed feedback. Remember to respond with ONLY valid JSON.`
          }
        ],
        temperature: 0.3,
        response_format: { type: "json_object" }
      });

      const responseContent = completion.choices?.[0]?.message?.content?.trim() || '{}';

      let ieltsResult;
      try {
        ieltsResult = JSON.parse(responseContent);
      } catch (parseError) {
        console.error('Failed to parse IELTS feedback JSON:', parseError);
        // Return a default structure if parsing fails
        ieltsResult = {
          overallBand: 0,
          scores: {
            fluencyCoherence: 0,
            lexicalResource: 0,
            grammaticalRange: 0,
            pronunciation: 0
          },
          feedback: {
            summary: 'Unable to analyze response. Please try again.',
            error: true
          }
        };
      }

      return res.status(200).json({
        ieltsScores: ieltsResult,
        transcript,
        metrics,
        question,
        part
      });

    } else {
      // Fallback: Handle text-only format (manual transcript analysis)
      const { text, question, part } = req.body || {};
      if (!text || typeof text !== 'string' || !text.trim()) {
        return res.status(400).json({ error: 'Missing transcript text or audio file' });
      }

      const completion = await client.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [
          {
            role: 'system',
            content: IELTS_EXAMINER_PROMPT
          },
          {
            role: 'user',
            content: `IELTS Speaking Part ${part || '1'} Question: "${question || 'General IELTS Question'}"

TRANSCRIPT OF RESPONSE:
"${text.trim()}"

Note: This is a text-only analysis without audio metrics. For pronunciation, provide general guidance based on written patterns (word choice that might be difficult to pronounce, etc.).

Analyze this IELTS speaking response and provide band scores with detailed feedback. Remember to respond with ONLY valid JSON.`
          }
        ],
        temperature: 0.3,
        response_format: { type: "json_object" }
      });

      const responseContent = completion.choices?.[0]?.message?.content?.trim() || '{}';

      let ieltsResult;
      try {
        ieltsResult = JSON.parse(responseContent);
      } catch (parseError) {
        console.error('Failed to parse IELTS feedback JSON:', parseError);
        ieltsResult = {
          overallBand: 0,
          scores: {
            fluencyCoherence: 0,
            lexicalResource: 0,
            grammaticalRange: 0,
            pronunciation: 0
          },
          feedback: {
            summary: 'Unable to analyze response. Please try again.',
            error: true
          }
        };
      }

      return res.status(200).json({
        ieltsScores: ieltsResult,
        transcript: text.trim(),
        question: question || 'General IELTS Question',
        part: part || '1'
      });
    }

  } catch (error) {
    console.error('Feedback generation failed:', error);
    console.error('Error details:', error.message);
    console.error('Error stack:', error.stack);

    // Check if it's an OpenAI API error
    if (error.status) {
      console.error('OpenAI API Status:', error.status);
      console.error('OpenAI API Error:', error.error);
      return res.status(error.status).json({
        error: 'OpenAI API error',
        details: error.message,
        status: error.status
      });
    }

    // Check if API key is missing
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: 'OpenAI API key not configured',
        details: 'OPENAI_API_KEY environment variable is missing'
      });
    }

    return res.status(500).json({
      error: 'Failed to generate feedback',
      details: error.message
    });
  }
}
