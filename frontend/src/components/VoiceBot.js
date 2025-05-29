import React, { useState, useRef } from "react";

const VoiceBot = () => {
  const [userText, setUserText] = useState("");
  const [botResponse, setBotResponse] = useState("");
  const [listening, setListening] = useState(false);
  const [manuallyStopped, setManuallyStopped] = useState(false);
  const synth = window.speechSynthesis;
  const recognitionRef = useRef(null);

    const startListening = () => {
      setManuallyStopped(false); // Reset stop flag
      const recognition = new window.webkitSpeechRecognition();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
    
      recognition.onstart = () => setListening(true);
    
      recognition.onresult = async (event) => {
        if(manuallyStopped) {
          return; // ❌ Don't process if user clicked stop
        }
    
        const spokenText = event.results[0][0].transcript;
        setUserText(spokenText);
    
        const reply = await getBotReply(spokenText);
        setBotResponse(reply);
        speak(reply);
        setListening(false);
      };
    
      recognition.onerror = (e) => {
        console.error("Speech Recognition Error:", e);
        setListening(false);
      };
    
      recognition.onend = () => setListening(false);
    
      recognition.start();
      recognitionRef.current = recognition;
    };
    
    const stopListening = () => {
      setManuallyStopped(true); // ✅ Mark it was stopped by user
      if (recognitionRef.current) {
        recognitionRef.current.stop();
        setListening(false);
      }
    };
    

  const getBotReply = async (userMessage) => {
    try {
      const res = await fetch("http://127.0.0.1:5000/chat", {  // 🔹 Change API to Flask
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userMessage }),
      });

      const data = await res.json();

      if (!data.reply) {
        return "Sorry, I couldn't understand that.";
      }

      return data.reply;
    } catch (error) {
      console.error("API Error:", error);
      return "Something went wrong while contacting Gemini AI.";
    }
  };

  const speak = (text) => {
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "en-US";
    utter.rate = 1;
    synth.speak(utter);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-white flex flex-col items-center justify-center p-6">
      <h1 className="text-4xl font-extrabold text-blue-900 mb-6">🎤 Voice Chatbot</h1>

      <div className="flex gap-4 mb-8">
        <button
          onClick={startListening}
          className="bg-blue-700 text-white px-6 py-3 rounded-full hover:bg-blue-800 transition"
          disabled={listening}
        >
          Start Talking 🎙️
        </button>
        <button
          onClick={stopListening}
          className="bg-red-600 text-white px-6 py-3 rounded-full hover:bg-red-700 transition"
          disabled={!listening}
        >
          Stop ❌
        </button>
      </div>

      <div className="bg-white p-6 rounded-xl shadow-lg w-full max-w-2xl text-center">
        <p className="text-gray-800 mb-2">
          <strong>You said:</strong> {userText || "Nothing yet..."}
        </p>
        <p className="text-green-700">
          <strong>Bot replied:</strong> {botResponse || "Waiting for input..."}
        </p>
      </div>
    </div>
  );
};

export default VoiceBot;
