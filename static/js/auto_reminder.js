// Auto Reminder JavaScript

// Function to check for active reminders
function checkActiveReminders() {
    fetch('/check_active_reminders')
        .then(response => response.json())
        .then(data => {
            const activeReminders = data.active_reminders;
            if (activeReminders && activeReminders.length > 0) {
                // Play each active reminder
                activeReminders.forEach(reminder => {
                    // Show browser notification
                    showNotification(reminder);
                    
                    // Play audio
                    if (reminder.audio_file) {
                        playReminderAudio(reminder.audio_file);
                    }
                });
            }
        })
        .catch(error => console.error('Error checking for active reminders:', error));
}

// Function to show browser notification
function showNotification(reminder) {
    // Check if browser supports notifications
    if (!('Notification' in window)) {
        console.error('This browser does not support desktop notification');
        return;
    }
    
    // Check if permission is already granted
    if (Notification.permission === 'granted') {
        createNotification(reminder);
    }
    // Otherwise, request permission
    else if (Notification.permission !== 'denied') {
        Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
                createNotification(reminder);
            }
        });
    }
}

// Function to create and display notification
function createNotification(reminder) {
    const title = 'Medicine Reminder';
    const options = {
        body: `It's time to take ${reminder.dosage} of ${reminder.medicine} ${reminder.method}`,
        icon: '/favicon.ico', // You can add a favicon to your app
        vibrate: [200, 100, 200]
    };
    
    const notification = new Notification(title, options);
    
    // Close notification after 10 seconds
    setTimeout(() => {
        notification.close();
    }, 10000);
}

// Function to play reminder audio
function playReminderAudio(audioFile) {
    const audio = new Audio('/reminders/' + audioFile);
    audio.play();
}

// Request notification permission when page loads
document.addEventListener('DOMContentLoaded', function() {
    if ('Notification' in window && Notification.permission !== 'granted' && Notification.permission !== 'denied') {
        Notification.requestPermission();
    }
    
    // Start checking for active reminders every 30 seconds
    setInterval(checkActiveReminders, 30000);
    
    // Also check immediately when page loads
    checkActiveReminders();
});