import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// Replace with your Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyA02JhNdktqa0n-N3R1tkqVPn8BUSYJ1Rc",
  authDomain: "inf-project-3710d.firebaseapp.com",
  projectId: "inf-project-3710d",
  storageBucket: "inf-project-3710d.firebasestorage.app",
  messagingSenderId: "175026727253",
  appId: "1:175026727253:web:fbb35c736d54d6ca5b67dc",
  measurementId: "G-2X57FQLD9T"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const googleProvider = new GoogleAuthProvider();

