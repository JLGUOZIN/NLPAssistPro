class DialogManager:
    def __init__(self):
        self.user_context = {}

    def update_context(self, user_id, intent, entities):
        if user_id not in self.user_context:
            self.user_context[user_id] = {'intents': [], 'entities': {}}
        self.user_context[user_id]['intents'].append(intent)
        self.user_context[user_id]['entities'].update(entities)

    def get_context(self, user_id):
        return self.user_context.get(user_id, {'intents': [], 'entities': {}})
