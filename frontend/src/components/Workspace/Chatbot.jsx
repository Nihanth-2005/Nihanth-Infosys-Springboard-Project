import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import toast from "react-hot-toast";
import { motion, AnimatePresence } from "framer-motion";
import { Send, MessageSquare, Bot, User } from "lucide-react";
import { API_ENDPOINTS } from "../../config/api";

function Chatbot({ workspaceId }) {
  const [messages, setMessages] = useState([
    {
      role: "bot",
      content:
        "Hello! I'm your AI assistant. How can I help you with your workspace today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(API_ENDPOINTS.CHATBOT, {
        message: input,
        workspace_id: workspaceId,
      });

      setMessages((prev) => [
        ...prev,
        { role: "bot", content: response.data.reply },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      toast.error("Failed to get response from chatbot");
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: "Sorry, I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div
      className="min-h-screen w-full bg-cover bg-center bg-fixed p-8"
      style={{
        backgroundImage:
          "url('https://images.stockcake.com/public/5/d/0/5d01f4cb-86f4-4bb1-83c8-166f3c5db1a1_large/neural-network-glow-stockcake.jpg')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundColor: "rgba(0, 0, 0, 0.85)",
        backgroundBlendMode: "overlay",
      }}
    >
      {/* Chat container */}
      <div className="max-w-5xl mx-auto rounded-2xl p-8 shadow-2xl border border-slate-800 bg-slate-900/70 backdrop-blur-md flex flex-col h-[80vh]">
        {/* Header */}
        <h2 className="text-2xl font-bold mb-6 text-slate-100 flex items-center gap-3">
          <MessageSquare className="w-6 h-6 text-blue-400" />
          AI Chatbot
        </h2>

        {/* Chat area with visible background */}
        <div
          className="flex-1 overflow-y-auto mb-4 space-y-4 pr-2 rounded-2xl p-6 border border-slate-700/50 shadow-inner backdrop-blur-sm"
          style={{
            backgroundImage:
              "url('https://images.stockcake.com/public/5/d/0/5d01f4cb-86f4-4bb1-83c8-166f3c5db1a1_large/neural-network-glow-stockcake.jpg')",
            backgroundSize: "cover",
            backgroundPosition: "center",
            backgroundAttachment: "fixed",
            backgroundColor: "rgba(0,0,0,0.45)",
            backgroundBlendMode: "overlay",
          }}
        >
          <AnimatePresence>
            {messages.map((message, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className={`flex gap-3 ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {message.role === "bot" && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-800/30">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                )}
                <div
                  className={`max-w-[70%] rounded-2xl px-4 py-3 shadow-md ${
                    message.role === "user"
                      ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white"
                      : "bg-slate-900/70 text-slate-100 border border-slate-700/70 backdrop-blur-sm"
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap leading-relaxed">
                    {message.content}
                  </p>
                </div>
                {message.role === "user" && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center">
                    <User className="w-5 h-5 text-white" />
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Typing indicator */}
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex gap-3"
            >
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-800/30">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-slate-900/70 rounded-2xl px-4 py-3 border border-slate-700 backdrop-blur-sm">
                <div className="flex gap-1">
                  <div
                    className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  ></div>
                </div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="flex gap-3 mt-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="flex-1 px-4 py-3 rounded-lg border border-slate-700 bg-slate-800/60 text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 backdrop-blur-sm"
            disabled={loading}
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Send className="w-5 h-5" />
          </motion.button>
        </div>
      </div>
    </div>
  );
}

export default Chatbot;
